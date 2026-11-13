# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-13 (core/models): ``DecisionsModel`` reuses one connection.

The ``decide`` tool keeps a single adapter for the agent's lifetime and
calls it for every routing decision, so each call used to pay a fresh
TCP+TLS handshake through module-level ``requests.post``.  The adapter
now posts through its own pooled ``httpx.Client``: against a real HTTP/1.1
keep-alive server, consecutive calls arrive over the same client socket.
"""

from __future__ import annotations

import gc
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.decisions_model import DecisionsModel, noul

_ANSWERS = {"damaged": {"probability": 0.97}}


class _KeepAliveHandler(BaseHTTPRequestHandler):
    """HTTP/1.1 decisions endpoint recording each request's client port and each hang-up."""

    protocol_version = "HTTP/1.1"
    client_ports: list[int] = []
    hung_up = threading.Semaphore(0)

    def finish(self) -> None:  # the client closed the connection
        super().finish()
        _KeepAliveHandler.hung_up.release()

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
    _KeepAliveHandler.hung_up = hung_up = threading.Semaphore(0)
    name, url = "openrouter/~typesafe/jev-latest", f"http://127.0.0.1:{server.server_port}/api"
    # Audit 2026-11-10 (A/C4-F5): the adapter closes its connection when it is
    # dropped (it used to stay open until a cyclic collection) or closed.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        m = DecisionsModel(name, base_url=url, api_key="k", model_config={"timeout": 5})
        questions = {"damaged": noul("Is the item damaged?")}
        for _ in range(3):
            assert m.decide("The package arrived crushed.", questions)["answers"] == _ANSWERS
        del m
        assert hung_up.acquire(timeout=10), "a dropped adapter kept its connection open"
        kept = DecisionsModel(name, base_url=url, api_key="k", model_config={"timeout": 5})
        assert kept.decide("The package arrived crushed.", questions)["answers"] == _ANSWERS
        kept.close()
        kept.close()  # idempotent
        assert hung_up.acquire(timeout=10), "close() left the connection open"
        with pytest.raises(KISSError, match="jev-latest is closed"):
            kept.decide("The package arrived crushed.", questions)
    finally:
        if gc_was_enabled:
            gc.enable()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert len(_KeepAliveHandler.client_ports) == 4
    assert len(set(_KeepAliveHandler.client_ports[:3])) == 1, _KeepAliveHandler.client_ports
