# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Local stand-in for the whatsapp-mcp bridge REST API (POST-only ``/api/send``).

``with bridge_server({"success": True, "message": "sent"}) as server:`` runs
the fake on an ephemeral port; ``server.server_address[1]`` is the port to
pass as ``bridge_port``.  Every POST is recorded in ``server.requests``
(see ``JsonWebhookServer``); every other verb answers 405 exactly like the
Go bridge's mux, so a tool that stops using POST fails its test.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from kiss.tests.agents.third_party_agents.recording_http import (
    JsonWebhookHandler,
    JsonWebhookServer,
    recording_server,
)


class _BridgeHandler(JsonWebhookHandler):
    def do_GET(self) -> None:  # noqa: N802
        """Refuse every verb but POST like the Go bridge's mux."""
        self._reply(405, "Method not allowed\n", "text/plain")

    do_PUT = do_PATCH = do_DELETE = do_GET  # noqa: N815


@contextmanager
def bridge_server(response_body: dict[str, Any]) -> Iterator[JsonWebhookServer]:
    """Run a bridge stand-in answering *response_body* for the block."""
    with recording_server(_BridgeHandler, JsonWebhookServer) as server:
        server.response_body = response_body
        yield server
