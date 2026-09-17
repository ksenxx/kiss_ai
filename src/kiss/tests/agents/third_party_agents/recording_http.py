# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared in-process HTTP test server that records every request.

Used by the Discord backend end-to-end tests (``test_typing_discord.py``
and ``test_send_failure_discord.py``): each test file supplies its own
``BaseHTTPRequestHandler`` that appends a request dict to
``RecordingServer.requests`` and serves Discord-shaped responses.
"""

from __future__ import annotations

from http.server import ThreadingHTTPServer
from typing import Any


class RecordingServer(ThreadingHTTPServer):
    """HTTP server that records every request it handles."""

    def __init__(self, address: tuple[str, int], handler: type) -> None:
        super().__init__(address, handler)
        self.requests: list[dict[str, Any]] = []
