# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A scripted local stand-in for the Anthropic Messages API.

Streaming requests (``"stream": true``, the agent's normal steps) get the
next scripted message as server-sent events; non-streaming requests (the
prompt-cache keep-alive pings) get a fixed one-word JSON reply, or an error
status when the script asks for one.  Every request body is recorded so a
test can check what the client sent.  Point the SDK at it with the
``ANTHROPIC_BASE_URL`` environment variable.

A scripted turn may carry ``thinking``, the text of its thinking block, and
``output_tokens``, what the reply is billed for (30 by default).

Served ``as_provider``, the stand-in also does what the provider does where
the cost of a session depends on it (:mod:`kiss.tests.core.local_anthropic_provider`):
the usage of a reply comes from a prompt cache of the requests, a reply to
a thinking request starts with a thinking block signed over the
conversation before it, a request that sends such a block back after that
conversation changed gets the provider's 400, and a ``clear_tool_uses``
edit is applied.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from kiss.tests.core import local_anthropic_provider as provider

MODEL = "claude-fable-5-1"
PING_TEXT = "ok"
PING_USAGE = {
    "input_tokens": 40,
    "output_tokens": 3,
    "cache_read_input_tokens": 20_000,
    "cache_creation_input_tokens": 0,
}


def tool_use_message(
    name: str,
    arguments: dict[str, Any],
    prompt_tokens: int,
    call_id: str,
    applied_edits: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Describe one scripted assistant turn that calls *name* with *arguments*.

    Args:
        name: The tool to call.
        arguments: Its input.
        prompt_tokens: Reported (uncached) input size.
        call_id: The ``tool_use`` id.
        applied_edits: The server-side context edits the turn reports in
            its final ``message_delta`` (``context_management.applied_edits``).
    """
    turn = {"name": name, "arguments": arguments, "prompt_tokens": prompt_tokens, "id": call_id}
    if applied_edits is not None:
        turn["applied_edits"] = applied_edits
    return turn


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _error_body(message: str) -> bytes:
    error = {"type": "invalid_request_error", "message": message}
    return json.dumps({"type": "error", "error": error}).encode()


def _thinking_events(signature: str, text: str) -> list[bytes]:
    """The events of a thinking block at index 0 that holds *text* and carries *signature*."""
    block = {"type": "thinking", "thinking": "", "signature": ""}
    deltas = [
        {"type": "thinking_delta", "thinking": text},
        {"type": "signature_delta", "signature": signature},
    ]
    return [
        _sse(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": block},
        ),
        *(
            _sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": d})
            for d in deltas
        ),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ]


def _stream_body(turn: dict[str, Any]) -> bytes:
    usage = turn.get("usage") or {
        "input_tokens": turn["prompt_tokens"],
        "output_tokens": 1,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    # The tool call follows the thinking block of a signed turn.
    index = 1 if "signature" in turn else 0
    message: dict[str, Any] = {
        "id": f"msg_{turn['id']}",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": usage,
    }
    block = {"type": "tool_use", "id": turn["id"], "name": turn["name"], "input": {}}
    delta: dict[str, Any] = {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"output_tokens": turn.get("output_tokens", 30)},
    }
    if "applied_edits" in turn:
        delta["context_management"] = {"applied_edits": turn["applied_edits"]}
    return b"".join(
        [
            _sse("message_start", {"type": "message_start", "message": message}),
            *(
                _thinking_events(
                    turn["signature"], turn.get("thinking", "Considering the next call.")
                )
                if index
                else []
            ),
            _sse(
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": block},
            ),
            _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": turn.get("raw_json") or json.dumps(turn["arguments"]),
                    },
                },
            ),
            _sse("content_block_stop", {"type": "content_block_stop", "index": index}),
            _sse("message_delta", delta),
            _sse("message_stop", {"type": "message_stop"}),
        ]
    )


def _ping_body() -> bytes:
    return json.dumps(
        {
            "id": "msg_ping",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [{"type": "text", "text": PING_TEXT}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": PING_USAGE,
        }
    ).encode()


class ScriptedHandler(BaseHTTPRequestHandler):
    """Answer streaming POSTs from the script (the last turn repeats) and pings with ``ok``."""

    script: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    ping_status: int = 200
    cache: provider.PromptCache | None = None
    lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        # The beta header of the request, recorded next to its body.
        body["_beta"] = self.headers.get("anthropic-beta", "")
        with self.lock:
            self.requests.append(body)
            streaming = bool(body.get("stream"))
            n_streamed = sum(1 for r in self.requests if r.get("stream"))
        if not streaming:
            if self.ping_status != 200:
                payload = _error_body("scripted ping failure")
                self._reply(self.ping_status, "application/json", payload)
                return
            self._reply(200, "application/json", _ping_body())
            return
        turn = self.script[min(n_streamed, len(self.script)) - 1]
        if self.cache is not None:
            # The check reads what the client sent, the cache what the model reads.
            error = provider.stale_thinking(body)
            if error is not None:
                self._reply(400, "application/json", _error_body(error))
                return
            read, applied = provider.clear_tool_uses(body, n_streamed)
            with self.lock:
                turn = {**turn, "usage": self.cache.usage(read), "applied_edits": applied}
            if (body.get("thinking") or {}).get("type") in ("enabled", "adaptive"):
                turn["signature"] = provider.signatures(body)[-1]
        self._reply(200, "text/event-stream", _stream_body(turn))

    def _reply(self, status: int, content_type: str, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


@contextmanager
def serve(
    script: list[dict[str, Any]], ping_status: int = 200, as_provider: bool = False
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Serve *script* on a free localhost port; yield ``(base_url, recorded_requests)``.

    Args:
        script: Scripted assistant turns from :func:`tool_use_message`.
        ping_status: HTTP status returned to non-streaming (keep-alive) requests.
        as_provider: Answer the streaming requests as the provider would (see the
            module docstring); the ``prompt_tokens`` of the script are then unused.
    """
    recorded: list[dict[str, Any]] = []

    class Handler(ScriptedHandler):
        pass

    Handler.script = script
    Handler.requests = recorded
    Handler.ping_status = ping_status
    Handler.cache = provider.PromptCache() if as_provider else None
    Handler.lock = threading.Lock()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", recorded
    finally:
        server.shutdown()
        server.server_close()
