# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``shutdown_pool_sockets`` walks a pool nobody made abortable and shuts its sockets down.

A pool given an ``AbortableNetworkBackend`` is shut down by the
backend's own ``abort`` alone (it kept every stream the pool's
connections hold); that case is in ``test_abortable_network_backend.py``.
Any other pool is walked connection by connection: here a plain httpx
client's kept-alive connection to a real local server.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from kiss.core.models.stream_abort import shutdown_pool_sockets


class _KeepAlive(BaseHTTPRequestHandler):
    """Answers every GET with ``ok`` on a kept-alive HTTP/1.1 connection, noting its port."""

    protocol_version = "HTTP/1.1"
    ports: list[int] = []

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self.ports.append(self.client_address[1])
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class _DaemonServer(ThreadingHTTPServer):
    daemon_threads = True


def test_a_plain_pools_kept_alive_socket_is_shut_down() -> None:
    """The pooled connection is shut down, so the next request needs a new one."""
    ports: list[int] = []
    server = _DaemonServer(("127.0.0.1", 0), type("_Handler", (_KeepAlive,), {"ports": ports}))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/"
    client = httpx.Client(trust_env=False)
    try:
        assert client.get(url).text == "ok"
        assert client.get(url).text == "ok"
        assert ports[0] == ports[1], "the connection was not kept alive"

        assert shutdown_pool_sockets(client) is True

        assert client.get(url).text == "ok"
        assert ports[2] != ports[0], "the shut-down connection was used again"
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(5)
