# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end checks of the shared test helpers fixed in the 2026-10-13 audit.

* ``recording_http.serving`` / ``start_serving`` poll the shutdown flag
  every 50 ms, so a fixture teardown no longer waits up to the stdlib's
  0.5 s default.
* ``recording_http.raw_post_status`` sends header lines as latin-1, the
  encoding ``http.server`` decodes them with.
* ``RecordingDaemon.close`` is idempotent.
* ``FakeComposioServer`` answers a per-poll status queue, refuses
  unknown API keys and undeletable accounts, and 404s unknown downloads.
"""

from __future__ import annotations

import time
from http.server import BaseHTTPRequestHandler
from typing import Any

import pytest
import requests

from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer
from kiss.tests.agents.third_party_agents.composio_test_utils import (
    API,
    API_KEY,
    fake_composio,
)
from kiss.tests.agents.third_party_agents.recording_daemon import RecordingDaemon
from kiss.tests.agents.third_party_agents.recording_http import (
    RecordingServer,
    raw_post_status,
    recording_server,
    serving,
)


class _HeaderRecorder(BaseHTTPRequestHandler):
    """Records the ``X-Probe`` header of every POST and answers 204."""

    server: RecordingServer  # type: ignore[assignment]

    def do_POST(self) -> None:  # noqa: N802
        self.server.requests.append({"probe": self.headers.get("X-Probe", "")})
        self.send_response(204)
        self.end_headers()

    def log_message(self, *_args: Any) -> None:  # type: ignore[override]
        """Silence request logging."""


def test_serving_teardown_does_not_wait_for_the_default_poll() -> None:
    """Ten start/stop cycles finish well inside ten of the stdlib's 0.5 s polls."""
    started = time.monotonic()
    for _ in range(10):
        with serving(ThreadedHTTPServer(("127.0.0.1", 0), _HeaderRecorder)):
            pass
    assert time.monotonic() - started < 1.0


def test_raw_post_status_sends_header_values_as_latin1() -> None:
    """A non-ASCII header value arrives as the one character written."""
    with recording_server(_HeaderRecorder) as server:
        assert raw_post_status(server.server_address[1], "X-Probe: \u00b2\r\n") == 204
    assert server.requests == [{"probe": "\u00b2"}]


def test_recording_daemon_close_is_idempotent() -> None:
    daemon = RecordingDaemon()
    daemon.close()
    daemon.close()
    assert not daemon.tmp.exists()


def test_fake_composio_status_queue_and_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    with fake_composio(monkeypatch) as composio:
        account = f"{composio.base_url}{API}/connected_accounts/ca_x"
        good = {"x-api-key": API_KEY}
        composio.set_statuses("ca_x", "INITIATED", "ACTIVE")
        polls = [requests.get(account, headers=good, timeout=10).json()["status"] for _ in range(3)]
        assert polls == ["INITIATED", "ACTIVE", "ACTIVE"]
        assert composio.paths("GET") == [f"{API}/connected_accounts/ca_x"] * 3

        assert requests.get(account, headers={"x-api-key": "other"}, timeout=10).status_code == 401
        composio.api_keys.add("other")
        assert requests.get(account, headers={"x-api-key": "other"}, timeout=10).status_code == 200

        composio.undeletable.add("ca_x")
        assert requests.delete(account, headers=good, timeout=10).status_code == 404
        composio.undeletable.clear()
        assert requests.delete(account, headers=good, timeout=10).status_code == 200
        assert composio.deleted == ["ca_x"] and "ca_x" not in composio.accounts

        assert requests.get(f"{composio.base_url}/download/0", timeout=10).status_code == 404
        composio.downloads.append(("text/plain", b"hi"))
        assert requests.get(f"{composio.base_url}/download/0", timeout=10).content == b"hi"
