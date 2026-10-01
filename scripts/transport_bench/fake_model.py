# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Stand-in OpenAI-compatible model for the transport benchmark.

Runs as its own process so its CPU time never lands in the daemon's or
the client's accounting.  Every chat-completions request is answered
with a server-sent-event stream of ``--chunks`` text deltas (paced by
``--gap-us``; both overridable per request with ``chunks=N`` /
``gap_us=N`` in the prompt text) followed by a ``finish`` tool call, so
the daemon's agent streams that many ``text_delta`` events to its clients
and then ends the task in one model round trip.

Usage::

    python fake_model.py --port 0 --chunks 2000 --chunk-chars 40 --gap-us 0

Prints ``READY <port>`` on stdout once it is listening.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MODEL = "gpt-4o-mini"
_WORDS = "the quick brown fox jumps over the lazy dog and keeps running ".split()


def _frame(payload: dict[str, Any]) -> bytes:
    """Encode one SSE ``data:`` frame of a chat-completion chunk."""
    base = {"id": "chatcmpl-bench", "object": "chat.completion.chunk",
            "created": 0, "model": MODEL}
    return b"data: " + json.dumps({**base, **payload}).encode() + b"\n\n"


def _text_chunk(text: str) -> bytes:
    return _frame({"choices": [{"index": 0, "delta": {"content": text},
                                "finish_reason": None}], "usage": None})


def _finish_frames(summary: str) -> list[bytes]:
    """The streamed ``finish`` tool call that ends the agent's run."""
    args = json.dumps({"success": "true", "summary_in_html": summary})
    return [
        _frame({"choices": [{"index": 0, "delta": {
            "role": "assistant", "content": "",
            "tool_calls": [{"index": 0, "id": "call_finish", "type": "function",
                            "function": {"name": "finish", "arguments": ""}}],
        }, "finish_reason": None}], "usage": None}),
        _frame({"choices": [{"index": 0, "delta": {
            "tool_calls": [{"index": 0, "function": {"arguments": args}}],
        }, "finish_reason": None}], "usage": None}),
        _frame({"choices": [{"index": 0, "delta": {},
                             "finish_reason": "tool_calls"}], "usage": None}),
        _frame({"choices": [], "usage": {"prompt_tokens": 10,
                                         "completion_tokens": 5,
                                         "total_tokens": 15}}),
        b"data: [DONE]\n\n",
    ]


def _overrides(body: str, chunks: int, gap_us: int) -> tuple[int, int]:
    """Read ``chunks=N`` / ``gap_us=N`` from the prompt text, if present."""
    for m in re.finditer(r"\b(chunks|gap_us)=(\d+)", body):
        if m.group(1) == "chunks":
            chunks = int(m.group(2))
        else:
            gap_us = int(m.group(2))
    return chunks, gap_us


class _Handler(BaseHTTPRequestHandler):
    """Stream ``chunks`` text deltas, then a ``finish`` tool call."""

    chunks = 100
    chunk_chars = 40
    gap_us = 0

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the access log."""

    def do_GET(self) -> None:  # noqa: N802
        """Answer ``/models`` so client-side catalog probes succeed."""
        body = json.dumps({"data": [{"id": MODEL, "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        """Stream the scripted completion."""
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode(errors="replace")
        chunks, gap_us = _overrides(body, self.chunks, self.gap_us)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        text = " ".join(_WORDS * (self.chunk_chars // 4 + 1))[: self.chunk_chars]
        for _ in range(chunks):
            self.wfile.write(_text_chunk(text))
            self.wfile.flush()
            if gap_us:
                time.sleep(gap_us / 1e6)
        for frame in _finish_frames("<p>bench done</p>"):
            self.wfile.write(frame)
        self.wfile.flush()


def main() -> None:
    """Serve the stand-in model until killed."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--chunks", type=int, default=100)
    ap.add_argument("--chunk-chars", type=int, default=40)
    ap.add_argument("--gap-us", type=int, default=0)
    ns = ap.parse_args()
    _Handler.chunks = ns.chunks
    _Handler.chunk_chars = ns.chunk_chars
    _Handler.gap_us = ns.gap_us
    httpd = ThreadingHTTPServer(("127.0.0.1", ns.port), _Handler)
    print(f"READY {httpd.server_port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
