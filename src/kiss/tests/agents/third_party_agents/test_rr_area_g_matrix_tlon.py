# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the matrix poll cursor fix (G-RC4) and tlon stub (G-R5).

matrix: ``poll_messages`` used to sync with ``since=self._next_batch``,
which is empty in every fresh cron-tick process, re-delivering recent
room timelines each tick.  It must now prefer the runner's persisted
cursor (the ``oldest`` argument).  Tested with a REAL ``nio``
AsyncClient syncing against a local HTTP server that records the
``since`` query parameter — no mocks or test doubles.

tlon: the backend carried a ``_event_queue`` that nothing ever filled
and an ``_sse_thread`` that was never started, so poll mode silently
yielded zero messages while looking implemented.  ``poll_messages`` is
now an explicit unsupported stub.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from importlib.util import find_spec
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from kiss.agents.third_party_agents._backend_utils import (
    ThreadedHTTPServer,
    stop_http_server,
)
from kiss.agents.third_party_agents.matrix_agent import MatrixChannelBackend
from kiss.agents.third_party_agents.tlon_agent import TlonChannelBackend


class _SyncRecordingHandler(BaseHTTPRequestHandler):
    """Records each sync request's query params and answers a minimal sync.

    Error modes (reset by the fixture): a request whose ``since`` equals
    ``REJECT_TOKEN`` is answered with a Matrix error body (no
    ``next_batch``), modelling a homeserver rejecting a stale persisted
    cursor; with ``fail_all`` set every request errors, modelling a
    server outage where even the full-sync retry fails.
    """

    requests: list[dict[str, list[str]]] = []
    REJECT_TOKEN = "rejected-token"
    fail_all = False

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        """Record the query string and reply with sync payload or error."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        type(self).requests.append(params)
        if type(self).fail_all or params.get("since") == [type(self).REJECT_TOKEN]:
            body = json.dumps(
                {"errcode": "M_UNKNOWN", "error": "Unrecognised sync token"}
            ).encode("utf-8")
            self.send_response(400)
        else:
            body = json.dumps(
                {
                    "next_batch": f"nb-{len(type(self).requests)}",
                    "rooms": {"join": {}, "invite": {}, "leave": {}},
                    "presence": {"events": []},
                    "account_data": {"events": []},
                    "to_device": {"events": []},
                    "device_lists": {"changed": [], "left": []},
                    "device_one_time_keys_count": {},
                }
            ).encode("utf-8")
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence request logging."""


@pytest.fixture()
def matrix_backend() -> Iterator[tuple[MatrixChannelBackend, list[dict[str, list[str]]]]]:
    """A matrix backend whose real nio client talks to a local sync server."""
    from nio import AsyncClient  # pyright: ignore[reportMissingImports]

    _SyncRecordingHandler.requests = []
    _SyncRecordingHandler.fail_all = False
    server = ThreadedHTTPServer(("127.0.0.1", 0), _SyncRecordingHandler)
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    backend = MatrixChannelBackend()
    client = AsyncClient(f"http://127.0.0.1:{port}", "@bot:example.org")
    # A real (locally issued) access token: nio refuses to sync while
    # logged out, and the recording server accepts any bearer token.
    client.access_token = "syt_local_test_token"
    client.user_id = "@bot:example.org"
    client.device_id = "TESTDEV"
    backend._client = client
    try:
        yield backend, _SyncRecordingHandler.requests
    finally:
        backend.disconnect()
        stop_http_server(server, thread)


@pytest.mark.skipif(find_spec("nio") is None, reason="matrix-nio is optional")
class TestMatrixSinceSelection:
    """poll_messages must sync from the persisted cursor when present.

    Skipif-guarded like the other optional-dependency channel tests
    (``matrix-nio`` is an optional package); the client-less guard is
    covered unconditionally by
    ``test_poll_without_client_returns_cursor_unchanged`` below.
    """

    def test_persisted_cursor_wins(
        self, matrix_backend: tuple[MatrixChannelBackend, list[dict[str, list[str]]]]
    ) -> None:
        """A real cursor in *oldest* is used even when _next_batch is set."""
        backend, requests = matrix_backend
        backend._next_batch = "stale-in-memory"
        messages, cursor = backend.poll_messages("!room:example.org", "persisted-42")
        assert requests[-1].get("since") == ["persisted-42"]
        assert messages == []
        # The server's next_batch becomes the new cursor for the runner.
        assert cursor == "nb-1"
        assert backend._next_batch == "nb-1"

    def test_default_cursor_zero_falls_back_to_next_batch(
        self, matrix_backend: tuple[MatrixChannelBackend, list[dict[str, list[str]]]]
    ) -> None:
        """The runner's fresh-state cursor '0' defers to the last sync token."""
        backend, requests = matrix_backend
        backend._next_batch = "mem-batch"
        backend.poll_messages("!room:example.org", "0")
        assert requests[-1].get("since") == ["mem-batch"]

    def test_no_cursor_and_no_batch_syncs_from_scratch(
        self, matrix_backend: tuple[MatrixChannelBackend, list[dict[str, list[str]]]]
    ) -> None:
        """With neither cursor nor batch, the initial sync has no since."""
        backend, requests = matrix_backend
        backend.poll_messages("!room:example.org", "")
        assert "since" not in requests[-1]

    def test_successive_polls_chain_next_batch(
        self, matrix_backend: tuple[MatrixChannelBackend, list[dict[str, list[str]]]]
    ) -> None:
        """A second poll without a persisted cursor uses the fresh token."""
        backend, requests = matrix_backend
        backend.poll_messages("!room:example.org", "")
        backend.poll_messages("!room:example.org", "0")
        assert requests[-1].get("since") == ["nb-1"]

    def test_rejected_since_token_recovers_with_full_sync(
        self, matrix_backend: tuple[MatrixChannelBackend, list[dict[str, list[str]]]]
    ) -> None:
        """A rejected persisted cursor triggers ONE since=None retry.

        The runner's nonempty-cursor guard keeps any old cursor when
        the poll returns an empty one, so without this recovery a
        rejected token would be resent (and rejected) forever.
        """
        backend, requests = matrix_backend
        messages, cursor = backend.poll_messages(
            "!room:example.org", _SyncRecordingHandler.REJECT_TOKEN
        )
        assert requests[-2].get("since") == [_SyncRecordingHandler.REJECT_TOKEN]
        assert "since" not in requests[-1]
        assert messages == []
        # The retry's fresh token replaces the rejected cursor.
        assert cursor == f"nb-{len(requests)}"
        assert backend._next_batch == cursor

    def test_rejected_token_with_failed_retry_keeps_old_cursor(
        self, matrix_backend: tuple[MatrixChannelBackend, list[dict[str, list[str]]]]
    ) -> None:
        """When the full-sync retry also fails, the tick is a no-op.

        Returning ``([], oldest)`` keeps the persisted cursor so a
        transient outage neither loses the cursor nor bricks the poll.
        """
        backend, requests = matrix_backend
        _SyncRecordingHandler.fail_all = True
        messages, cursor = backend.poll_messages(
            "!room:example.org", _SyncRecordingHandler.REJECT_TOKEN
        )
        assert messages == []
        assert cursor == _SyncRecordingHandler.REJECT_TOKEN
        # Exactly one retry: the rejected since, then the full sync.
        assert len(requests) == 2
        assert "since" not in requests[-1]


class TestMatrixWithoutClient:
    """The pre-sync guard needs no nio and no server."""

    def test_poll_without_client_returns_cursor_unchanged(self) -> None:
        """An unconnected backend polls to nothing, keeping the cursor."""
        backend = MatrixChannelBackend()
        assert backend.poll_messages("!room:example.org", "cur-1") == ([], "cur-1")


class TestTlonPollStub:
    """tlon poll mode is explicitly unsupported (no phantom event queue)."""

    def test_poll_returns_nothing_and_keeps_cursor(self) -> None:
        """The stub returns no messages and echoes the cursor."""
        backend = TlonChannelBackend()
        assert backend.poll_messages("group/chan", "cur-7", limit=5) == ([], "cur-7")
