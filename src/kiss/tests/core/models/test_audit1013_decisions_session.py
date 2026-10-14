# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-13 (core/models): ``DecisionsModel`` reuses one connection.

The ``decide`` tool keeps a single adapter for the agent's lifetime and
calls it for every routing decision, so each call used to pay a fresh
TCP+TLS handshake through module-level ``requests.post``.  The adapter
now posts through its own ``requests.Session``: against a real HTTP/1.1
keep-alive server, consecutive calls arrive over the same client socket.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from kiss.core.models.decisions_model import DecisionsModel, noul

_ANSWERS = {"damaged": {"probability": 0.97}}


class _KeepAliveHandler(BaseHTTPRequestHandler):
    """HTTP/1.1 decisions endpoint that records the client port of every request."""

    protocol_version = "HTTP/1.1"
    client_ports: list[int] = []

    def do_POST(self) -> None:  # noqa: N802 — http.server API
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        _KeepAliveHandler.client_ports.append(self.client_address[1])
        payload = json.dumps({"model": "typesafe/jev", "answers": _ANSWERS, "usage": {}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def test_consecutive_decide_calls_share_one_connection() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _KeepAliveHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _KeepAliveHandler.client_ports = []
    try:
        m = DecisionsModel(
            "openrouter/~typesafe/jev-latest",
            base_url=f"http://127.0.0.1:{server.server_port}/api",
            api_key="k",
            model_config={"timeout": 5},
        )
        questions = {"damaged": noul("Is the item damaged?")}
        for _ in range(3):
            assert m.decide("The package arrived crushed.", questions)["answers"] == _ANSWERS
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert len(_KeepAliveHandler.client_ports) == 3
    assert len(set(_KeepAliveHandler.client_ports)) == 1, _KeepAliveHandler.client_ports
