# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end checks of the shared test helpers fixed in the 2026-10-13 audit.

* ``recording_http.serving`` / ``start_serving`` poll the shutdown flag
  every 50 ms, so a fixture teardown no longer waits a whole 0.5 s
  stdlib poll.
* ``recording_http.raw_post_status`` sends header lines as latin-1, the
  encoding ``http.server`` decodes them with.
* ``RecordingDaemon.close`` is idempotent.
* ``FakeComposioServer`` answers a per-poll status queue, refuses
  unknown API keys and undeletable accounts, and 404s unknown downloads.
* ``EyreHandler`` and ``PhoneApiHandler`` answer their service's
  endpoints, 404 anything else, and record every request.
"""

from __future__ import annotations

import sys
import threading
import time

import pytest
import requests

from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer, stop_http_server
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
    start_serving,
)


class _HeaderRecorder(RecordingHandler):
    """Records the ``X-Probe`` header of every POST and answers 204."""

    server: RecordingServer  # type: ignore[assignment]

    def do_POST(self) -> None:  # noqa: N802
        self.record_request(None, probe=self.headers.get("X-Probe", ""))
        self.send_response(204)
        self.end_headers()


def test_serving_teardown_does_not_wait_for_the_default_poll() -> None:
    """Ten immediate teardowns finish well inside ten of the stdlib's 0.5 s polls.

    A teardown right after the start waits one whole poll: the serving
    thread is already blocked in ``select(poll_interval)`` when
    ``shutdown()`` raises its flag, and sees the flag only once that
    select times out.  Measured over ten cycles: 5.0 s with the stdlib's
    0.5 s poll, 0.5 s with the helper's 50 ms poll.  Only the ``stop``
    half of each cycle is timed (binding a port and starting a thread is
    the same either way); the 4 s bound gives the helper 8x headroom on
    a loaded machine and still fails the stdlib poll by a second.
    """
    stopping = 0.0
    for _ in range(10):
        server = ThreadedHTTPServer(("127.0.0.1", 0), _HeaderRecorder)
        thread = start_serving(server)
        started = time.monotonic()
        stop_http_server(server, thread)
        stopping += time.monotonic() - started
    assert stopping < 4.0


def test_raw_post_status_sends_header_values_as_latin1() -> None:
    """A non-ASCII header value arrives as the one character written."""
    with recording_server(_HeaderRecorder) as server:
        assert raw_post_status(server.server_address[1], "X-Probe: \u00b2\r\n") == 204
    assert [r["probe"] for r in server.requests] == ["\u00b2"]


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


_BLOB_CALLERS = "abcdefghijklmnop"
_BLOB_ROUNDS = 40


class _BlobServer(RecordingServer):
    """Upstream whose answers to one round of callers leave together."""

    def __init__(self, address: tuple[str, int], handler: type) -> None:
        super().__init__(address, handler)
        self.barrier = threading.Barrier(len(_BLOB_CALLERS))


class _BlobHandler(RecordingHandler):
    """Answers ``GET /<letter>`` with 64 bytes of that letter (a binary body)."""

    server: _BlobServer  # type: ignore[assignment]

    def do_GET(self) -> None:  # noqa: N802
        self.server.barrier.wait(timeout=30)  # the proxies then store their bodies at once
        self._reply(200, self.path[1:2].encode() * 64, "application/octet-stream")


def _proxy_blob(composio_url: str, upstream_url: str, letter: str, errors: list[str]) -> None:
    """Fetch ``/<letter>`` through the proxy every round; record every wrong download."""
    body = {"connected_account_id": "ca_x", "method": "GET", "endpoint": f"{upstream_url}/{letter}"}
    try:
        with requests.Session() as session:
            for _ in range(_BLOB_ROUNDS):
                answer = session.post(
                    f"{composio_url}{API}/tools/execute/proxy",
                    json=body,
                    headers={"x-api-key": API_KEY},
                    timeout=10,
                ).json()
                data = session.get(answer["binary_data"]["url"], timeout=10).content
                if data != letter.encode() * 64:
                    errors.append(f"{letter}: got {data[:1]!r} from {answer['binary_data']['url']}")
    except Exception as exc:  # noqa: BLE001 (surfaced on the test thread)
        errors.append(f"{letter}: {exc!r}")


def test_fake_composio_binary_proxy_names_each_callers_own_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent binary proxies never hand one caller another caller's bytes."""
    with (
        fake_composio(monkeypatch) as composio,
        recording_server(_BlobHandler, _BlobServer) as upstream,
    ):
        composio.accounts["ca_x"] = "ACTIVE"
        errors: list[str] = []
        threads = [
            threading.Thread(
                target=_proxy_blob,
                args=(composio.base_url, upstream.base_url, letter, errors),
                daemon=True,
            )
            for letter in _BLOB_CALLERS
        ]
        switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=120)
        finally:
            sys.setswitchinterval(switch_interval)
        assert not any(thread.is_alive() for thread in threads)
        assert errors == []
        assert len(composio.downloads) == len(_BLOB_CALLERS) * _BLOB_ROUNDS


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
        {"method": "GET", "path": path, "query": query, "authorization": "", "body": None}
        for path, query in [
            ("/api/device/info", {}),
            ("/api/sms/messages", {"since": ["7"], "q": [""]}),
            ("/api/nope", {}),
        ]
    ]
