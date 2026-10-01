# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Local stand-in for the whatsapp-mcp bridge REST API (POST-only ``/api/send``).

``with bridge_server({"success": True, "message": "sent"}) as server:`` runs
the fake on an ephemeral port; ``server.server_address[1]`` is the port to
pass as ``bridge_port``.  Every POST is recorded in ``server.requests`` as
``{"path", "json"}``; GET answers 405 exactly like the Go bridge's mux.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from typing import Any, cast

from kiss.tests.agents.third_party_agents.recording_http import RecordingServer, recording_server


class BridgeServer(RecordingServer):
    """Recording server answering every POST with the canned ``response_body``."""

    def __init__(self, address: tuple[str, int], handler: type) -> None:
        super().__init__(address, handler)
        self.response_body: dict[str, Any] = {}


class _BridgeHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self.send_response(405)
        self.end_headers()
        self.wfile.write(b"Method not allowed\n")

    def do_POST(self) -> None:  # noqa: N802
        server = cast(BridgeServer, self.server)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        server.requests.append({"path": self.path, "json": body})
        payload = json.dumps(server.response_body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:  # type: ignore[override]
        pass


@contextmanager
def bridge_server(response_body: dict[str, Any]) -> Iterator[BridgeServer]:
    """Run a bridge stand-in answering *response_body* for the block."""
    with recording_server(_BridgeHandler, BridgeServer) as server:
        server.response_body = response_body
        yield server
