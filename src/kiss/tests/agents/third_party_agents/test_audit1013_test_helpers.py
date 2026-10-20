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
* ``EyreHandler`` and ``PhoneApiHandler`` answer their service's
  endpoints, 404 anything else, and record every request.
"""

from __future__ import annotations

import time

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
    EyreHandler,
    PhoneApiHandler,
    RecordingHandler,
    RecordingServer,
    raw_post_status,
    recording_server,
    serving,
)


class _HeaderRecorder(RecordingHandler):
    """Records the ``X-Probe`` header of every POST and answers 204."""

    server: RecordingServer  # type: ignore[assignment]

    def do_POST(self) -> None:  # noqa: N802
        self.server.requests.append({"probe": self.headers.get("X-Probe", "")})
        self.send_response(204)
        self.end_headers()


def test_serving_teardown_does_not_wait_for_the_default_poll() -> None:
    """Ten start/stop cycles finish well inside ten of the stdlib's 0.5 s polls.

    With the stdlib default every idle teardown waits a whole 0.5 s poll,
    so ten cycles take about 5 s; the bound is half of that, which leaves
    room for a loaded machine without accepting the default poll.
    """
    stdlib_poll = 0.5
    started = time.monotonic()
    for _ in range(10):
        with serving(ThreadedHTTPServer(("127.0.0.1", 0), _HeaderRecorder)):
            pass
    assert time.monotonic() - started < 10 * stdlib_poll / 2


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


def test_eyre_handler_routes_and_records() -> None:
    with recording_server(EyreHandler) as server:
        base = server.base_url
        assert requests.post(f"{base}/~/login", data="password=x", timeout=10).status_code == 204
        assert (
            requests.put(f"{base}/~/channel/uid1", json=[{"id": 1}], timeout=10).status_code == 204
        )
        assert (
            requests.post(f"{base}/~/channel/uid1", json=[{"id": 2}], timeout=10).status_code == 204
        )
        scry = requests.get(f"{base}/~/scry/channels/channels/~zod/g/light.json", timeout=10)
        assert scry.status_code == 200 and scry.json() == {"channels": {}}
        for method in (requests.get, requests.put, requests.post):
            assert method(f"{base}/elsewhere", timeout=10).status_code == 404
    assert [(r["method"], r["path"], r["body"]) for r in server.requests[:3]] == [
        ("POST", "/~/login", "password=x"),
        ("PUT", "/~/channel/uid1", '[{"id": 1}]'),
        ("POST", "/~/channel/uid1", '[{"id": 2}]'),
    ]
    assert [r["method"] for r in server.requests[4:]] == ["GET", "PUT", "POST"]


class _OneMessagePhone(PhoneApiHandler):
    messages = [{"timestamp": 1, "from": "+1", "body": "hi", "id": "1"}]


def test_phone_api_handler_routes_and_records() -> None:
    with recording_server(_OneMessagePhone) as server:
        base = server.base_url
        assert requests.get(f"{base}/api/device/info", timeout=10).json() == {
            "device_name": "testphone"
        }
        polled = requests.get(f"{base}/api/sms/messages?since=7&q=", timeout=10).json()
        assert polled == {"messages": _OneMessagePhone.messages}
        assert requests.get(f"{base}/api/nope", timeout=10).status_code == 404
    assert server.requests == [
        {"method": "GET", "path": "/api/device/info", "query": {}},
        {"method": "GET", "path": "/api/sms/messages", "query": {"since": ["7"], "q": [""]}},
        {"method": "GET", "path": "/api/nope", "query": {}},
    ]
