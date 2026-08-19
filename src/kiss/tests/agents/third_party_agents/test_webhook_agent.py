# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the webhook channel agent — no mocks or test doubles.

Starts the real embedded HTTP server on an ephemeral port and exercises
signature verification, body caps, idempotency, rate limiting, payload
filters, template rendering, and config/route persistence with real
HTTP POSTs via ``requests``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import socket
import stat
import sys
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

import kiss.agents.third_party_agents.webhook_agent as webhook_agent_mod
from kiss.agents.third_party_agents.webhook_agent import (
    _RATE_LIMIT_EVENTS,
    _RATE_LIMIT_WINDOW_SECONDS,
    WebhookAgent,
    WebhookChannelBackend,
    _config,
)

_AUTH_TRIO = {"check_webhook_auth", "authenticate_webhook", "clear_webhook_auth"}
_SECRET = "test-secret-123"


@pytest.fixture(autouse=True)
def _fresh_webhook_config():
    """Start and end every test with no persisted webhook config."""
    _config.clear()
    yield
    _config.clear()


def _github_headers(body: bytes, delivery: str = "", secret: str = _SECRET) -> dict[str, str]:
    """Build valid GitHub-scheme signature headers for *body*."""
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    headers = {"X-Hub-Signature-256": f"sha256={digest}", "Content-Type": "application/json"}
    if delivery:
        headers["X-GitHub-Delivery"] = delivery
    return headers


def _generic_headers(
    body: bytes, ts: float | None = None, delivery: str = "", secret: str = _SECRET
) -> dict[str, str]:
    """Build valid generic-scheme signature headers for *body*."""
    ts_str = str(int(time.time() if ts is None else ts))
    digest = hmac.new(secret.encode(), f"{ts_str}.".encode() + body, hashlib.sha256).hexdigest()
    headers = {
        "X-Kiss-Timestamp": ts_str,
        "X-Kiss-Signature": digest,
        "Content-Type": "application/json",
    }
    if delivery:
        headers["X-Kiss-Delivery"] = delivery
    return headers


def _start_backend(**route_kwargs) -> tuple[WebhookChannelBackend, str]:
    """Authenticate on an ephemeral port, add one route, and start the server."""
    agent = WebhookAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    result = json.loads(tools["authenticate_webhook"](port="0"))
    assert result["ok"]
    backend = agent._backend
    added = json.loads(backend.add_webhook_route(**route_kwargs))
    assert added["ok"], added
    assert backend.connect(), backend.connection_info
    return backend, f"http://127.0.0.1:{backend._bound_port}"


def test_agent_unauthenticated_exposes_only_auth_trio() -> None:
    """A fresh agent is unauthenticated and offers exactly the auth trio."""
    agent = WebhookAgent()
    assert agent.name == "Webhook Agent"
    assert agent._is_authenticated() is False
    assert {t.__name__ for t in agent._get_tools()} == _AUTH_TRIO


def test_auth_trio_roundtrip_and_persistence() -> None:
    """authenticate persists 0600 config, check reports it, clear removes it."""
    agent = WebhookAgent()
    tools = {t.__name__: t for t in agent._get_tools()}

    unauth = tools["check_webhook_auth"]()
    assert "not configured" in unauth.lower()
    assert "port must be" in tools["authenticate_webhook"](port="abc")
    assert "port must be" in tools["authenticate_webhook"](port="70000")

    result = json.loads(tools["authenticate_webhook"]())
    assert result["ok"]
    path = _config.path
    assert path.exists()
    assert "third_party_agents/webhook" in str(path)
    if sys.platform != "win32":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    saved = json.loads(path.read_text())
    assert saved["port"] == "18090"
    assert json.loads(saved["routes"]) == {}

    checked = json.loads(tools["check_webhook_auth"]())
    assert checked["ok"] and checked["port"] == "18090"
    assert agent._is_authenticated()
    tool_names = {t.__name__ for t in agent._get_tools()}
    assert {"add_webhook_route", "remove_webhook_route", "list_webhook_routes"} <= tool_names

    assert "cleared" in tools["clear_webhook_auth"]().lower()
    assert not path.exists()
    assert agent._is_authenticated() is False


def test_module_get_tools_nonempty() -> None:
    """The module-level get_tools() tools-file contract returns tools."""
    tools = webhook_agent_mod.get_tools()
    assert tools
    assert _AUTH_TRIO <= {t.__name__ for t in tools}


def test_route_add_list_remove_persistence() -> None:
    """Routes persist in config and are visible to freshly built agents."""
    agent = WebhookAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    assert json.loads(tools["authenticate_webhook"](port="0"))["ok"]
    backend = agent._backend

    bad = json.loads(backend.add_webhook_route("bad name!", _SECRET))
    assert not bad["ok"]
    assert not json.loads(backend.add_webhook_route("r1", ""))["ok"]
    assert not json.loads(backend.add_webhook_route("r1", _SECRET, kind="weird"))["ok"]
    assert not json.loads(backend.add_webhook_route("r1", _SECRET, filters_json="[1]"))["ok"]

    added = json.loads(
        backend.add_webhook_route(
            "ci",
            _SECRET,
            kind="github",
            prompt_template="build {status}",
            filters_json='{"branch": "main"}',
        )
    )
    assert added["ok"] and added["path"] == "/hook/ci"

    listed = json.loads(backend.list_webhook_routes())
    assert listed["ok"]
    assert [r["name"] for r in listed["routes"]] == ["ci"]
    route = listed["routes"][0]
    assert route["kind"] == "github"
    assert route["filters"] == {"branch": "main"}
    assert "secret" not in route

    fresh = WebhookAgent()
    assert "ci" in fresh._backend._routes
    fresh_listed = json.loads(fresh._backend.list_webhook_routes())
    assert [r["name"] for r in fresh_listed["routes"]] == ["ci"]

    assert json.loads(backend.remove_webhook_route("ci"))["ok"]
    assert json.loads(backend.list_webhook_routes())["routes"] == []
    assert not json.loads(backend.remove_webhook_route("ci"))["ok"]
    persisted = _config.load()
    assert persisted is not None
    assert json.loads(persisted["routes"]) == {}


def test_connect_unconfigured_returns_false() -> None:
    """connect() fails cleanly when no config exists."""
    backend = WebhookChannelBackend()
    assert backend.connect() is False
    assert "no webhook config" in backend.connection_info.lower()


def test_github_valid_signature_accepted_and_rendered() -> None:
    """A correctly signed GitHub event is queued with the rendered template."""
    backend, base = _start_backend(
        name="gh",
        secret=_SECRET,
        kind="github",
        prompt_template="PR {action} by {sender.login} {unknown.thing} :: {payload}",
    )
    try:
        payload = {"action": "opened", "sender": {"login": "alice"}}
        body = json.dumps(payload).encode()
        resp = requests.post(
            f"{base}/hook/gh", data=body, headers=_github_headers(body), timeout=10
        )
        assert resp.status_code == 200
        messages, _ = backend.poll_messages("gh", "0")
        assert len(messages) == 1
        msg = messages[0]
        compact = json.dumps(payload, separators=(",", ":"))
        assert msg["text"] == f"PR opened by alice {{unknown.thing}} :: {compact}"
        assert msg["user"] == "gh" and msg["channel_id"] == "gh" and msg["ts"]
        assert backend.is_from_bot(msg) is False
    finally:
        backend.disconnect()


def test_github_wrong_signature_rejected_401() -> None:
    """Bad or missing GitHub signatures get 401 and nothing is queued."""
    backend, base = _start_backend(name="gh", secret=_SECRET, kind="github")
    try:
        body = b'{"x": 1}'
        bad = _github_headers(body, secret="wrong-secret")
        assert (
            requests.post(f"{base}/hook/gh", data=body, headers=bad, timeout=10).status_code == 401
        )
        assert requests.post(f"{base}/hook/gh", data=body, timeout=10).status_code == 401
        assert backend.poll_messages("gh", "0")[0] == []
    finally:
        backend.disconnect()


def test_generic_valid_and_stale_timestamp() -> None:
    """Generic scheme accepts fresh signed events and rejects stale timestamps."""
    backend, base = _start_backend(name="gen", secret=_SECRET, kind="generic")
    try:
        body = b'{"event": "ping"}'
        ok = requests.post(
            f"{base}/hook/gen", data=body, headers=_generic_headers(body), timeout=10
        )
        assert ok.status_code == 200
        assert len(backend.poll_messages("gen", "0")[0]) == 1

        stale = _generic_headers(body, ts=time.time() - 400)
        assert (
            requests.post(f"{base}/hook/gen", data=body, headers=stale, timeout=10).status_code
            == 401
        )
        no_ts = {"X-Kiss-Signature": "deadbeef"}
        assert (
            requests.post(f"{base}/hook/gen", data=body, headers=no_ts, timeout=10).status_code
            == 401
        )
        assert backend.poll_messages("gen", "0")[0] == []
    finally:
        backend.disconnect()


def test_oversized_body_rejected_413() -> None:
    """Bodies over 1 MB are rejected with 413 before any processing."""
    backend, base = _start_backend(name="big", secret=_SECRET, kind="github")
    try:
        body = b'{"pad": "' + b"a" * (1024 * 1024) + b'"}'
        resp = requests.post(
            f"{base}/hook/big", data=body, headers=_github_headers(body), timeout=30
        )
        assert resp.status_code == 413
        assert backend.poll_messages("big", "0")[0] == []
    finally:
        backend.disconnect()


def test_duplicate_delivery_id_dropped() -> None:
    """A replayed delivery id gets 200 but is queued only once."""
    backend, base = _start_backend(name="dup", secret=_SECRET, kind="github")
    try:
        body = b'{"n": 1}'
        headers = _github_headers(body, delivery="delivery-42")
        for _ in range(2):
            assert (
                requests.post(
                    f"{base}/hook/dup", data=body, headers=headers, timeout=10
                ).status_code
                == 200
            )
        assert len(backend.poll_messages("dup", "0")[0]) == 1
    finally:
        backend.disconnect()


def test_filter_mismatch_dropped() -> None:
    """Events failing a dot-path filter are dropped with 200; matches pass."""
    backend, base = _start_backend(
        name="flt",
        secret=_SECRET,
        kind="github",
        prompt_template="{action}",
        filters_json='{"action": "opened"}',
    )
    try:
        closed = json.dumps({"action": "closed"}).encode()
        resp = requests.post(
            f"{base}/hook/flt", data=closed, headers=_github_headers(closed), timeout=10
        )
        assert resp.status_code == 200
        assert backend.poll_messages("flt", "0")[0] == []

        opened = json.dumps({"action": "opened"}).encode()
        assert (
            requests.post(
                f"{base}/hook/flt", data=opened, headers=_github_headers(opened), timeout=10
            ).status_code
            == 200
        )
        messages, _ = backend.poll_messages("flt", "0")
        assert [m["text"] for m in messages] == ["opened"]
    finally:
        backend.disconnect()


def test_rate_limit_429_after_60_events() -> None:
    """The 61st accepted event within a minute is rejected with 429."""
    backend, base = _start_backend(name="rl", secret=_SECRET, kind="github")
    try:
        body = b'{"n": 1}'
        headers = _github_headers(body)
        with requests.Session() as session:
            for i in range(60):
                resp = session.post(f"{base}/hook/rl", data=body, headers=headers, timeout=10)
                assert resp.status_code == 200, f"event {i} -> {resp.status_code}"
            assert (
                session.post(f"{base}/hook/rl", data=body, headers=headers, timeout=10).status_code
                == 429
            )
        assert len(backend.poll_messages("rl", "0", limit=100)[0]) == 60
    finally:
        backend.disconnect()


def test_unknown_route_and_path_404() -> None:
    """Unknown route names and non-/hook/ paths return 404."""
    backend, base = _start_backend(name="known", secret=_SECRET, kind="github")
    try:
        body = b"{}"
        headers = _github_headers(body)
        assert (
            requests.post(f"{base}/hook/nope", data=body, headers=headers, timeout=10).status_code
            == 404
        )
        assert (
            requests.post(f"{base}/other", data=body, headers=headers, timeout=10).status_code
            == 404
        )
    finally:
        backend.disconnect()


def test_invalid_json_payload_400() -> None:
    """A correctly signed but non-JSON body is rejected with 400."""
    backend, base = _start_backend(name="jsn", secret=_SECRET, kind="github")
    try:
        body = b"not json"
        resp = requests.post(
            f"{base}/hook/jsn", data=body, headers=_github_headers(body), timeout=10
        )
        assert resp.status_code == 400
        assert backend.poll_messages("jsn", "0")[0] == []
    finally:
        backend.disconnect()


def test_poll_messages_filters_other_routes() -> None:
    """poll_messages(channel_id) discards messages from other routes."""
    backend, base = _start_backend(name="a", secret=_SECRET, kind="github")
    try:
        assert json.loads(backend.add_webhook_route("b", _SECRET, kind="github"))["ok"]
        for route in ("a", "b"):
            body = json.dumps({"route": route}).encode()
            assert (
                requests.post(
                    f"{base}/hook/{route}", data=body, headers=_github_headers(body), timeout=10
                ).status_code
                == 200
            )
        messages, _ = backend.poll_messages("b", "0")
        assert [m["channel_id"] for m in messages] == ["b"]
        assert backend.poll_messages("b", "0")[0] == []
    finally:
        backend.disconnect()


def _raw_post_status(port: int, path: str, headers: dict[str, str], body: bytes = b"") -> int:
    """POST over a raw socket (allows malformed/missing Content-Length)."""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        lines = [f"POST {path} HTTP/1.1", f"Host: 127.0.0.1:{port}", "Connection: close"]
        lines.extend(f"{k}: {v}" for k, v in headers.items())
        sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode() + body)
        data = b""
        while b"\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
        return int(data.split(b" ", 2)[1])


def _write_delivery_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str) -> str:
    """Create a real importable delivery module on sys.path; return its name."""
    name = f"webhook_delivery_target_{uuid.uuid4().hex}"
    (tmp_path / f"{name}.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


_CONTROLLABLE_DELIVERY_MODULE = """\
from pathlib import Path

_HERE = Path(__file__)
OUT = _HERE.with_suffix(".out")
FAIL_FLAG = _HERE.with_suffix(".fail")
LOG = _HERE.with_suffix(".log")


class _Backend:
    def connect(self):
        with LOG.open("a") as f:
            f.write("connect\\n")
        return True

    def send_message(self, channel_id, text, thread_ts=""):
        if FAIL_FLAG.exists():
            raise RuntimeError("delivery target down")
        with OUT.open("a") as f:
            f.write(f"{channel_id}|{text}\\n")

    def disconnect(self):
        with LOG.open("a") as f:
            f.write("disconnect\\n")


def _make_backend():
    return _Backend()
"""

_SYS_EXIT_DELIVERY_MODULE = """\
import sys


def _make_backend():
    print("Not configured.")
    sys.exit(1)
"""

_CONNECT_FALSE_DELIVERY_MODULE = """\
from pathlib import Path

LOG = Path(__file__).with_suffix(".log")


class _Backend:
    def connect(self):
        return False

    def send_message(self, channel_id, text, thread_ts=""):
        raise AssertionError("send_message must not run when connect() fails")

    def disconnect(self):
        with LOG.open("a") as f:
            f.write("disconnect\\n")


def _make_backend():
    return _Backend()
"""


def test_malformed_content_length_rejected_without_read() -> None:
    """Negative/non-decimal Content-Length -> 400, missing -> 411; server stays up."""
    backend, base = _start_backend(name="cl", secret=_SECRET, kind="github")
    try:
        port = backend._bound_port
        assert _raw_post_status(port, "/hook/cl", {"Content-Length": "-1"}) == 400
        assert _raw_post_status(port, "/hook/cl", {"Content-Length": "nope"}) == 400
        assert _raw_post_status(port, "/hook/cl", {"Content-Length": "+1"}, b"x") == 400
        assert _raw_post_status(port, "/hook/cl", {}) == 411
        assert backend.poll_messages("cl", "0")[0] == []

        body = b'{"after": "malformed"}'
        resp = requests.post(
            f"{base}/hook/cl", data=body, headers=_github_headers(body), timeout=10
        )
        assert resp.status_code == 200
        assert len(backend.poll_messages("cl", "0")[0]) == 1
    finally:
        backend.disconnect()


def test_same_delivery_id_on_two_routes_both_processed() -> None:
    """Idempotency is scoped per route: one delivery id may hit each route once."""
    backend, base = _start_backend(name="ra", secret=_SECRET, kind="github")
    try:
        assert json.loads(backend.add_webhook_route("rb", _SECRET, kind="github"))["ok"]
        body = b'{"n": 1}'
        headers = _github_headers(body, delivery="shared-id-1")
        for route in ("ra", "rb"):
            resp = requests.post(f"{base}/hook/{route}", data=body, headers=headers, timeout=10)
            assert resp.status_code == 200
        messages, _ = backend.poll_messages("", "0")
        assert sorted(m["channel_id"] for m in messages) == ["ra", "rb"]

        for route in ("ra", "rb"):
            resp = requests.post(f"{base}/hook/{route}", data=body, headers=headers, timeout=10)
            assert resp.status_code == 200
        assert backend.poll_messages("", "0")[0] == []
    finally:
        backend.disconnect()


def test_rate_limited_event_retry_is_not_dropped_as_duplicate() -> None:
    """A 429'd delivery id is not recorded as seen; its later retry is processed."""
    backend, base = _start_backend(name="rl2", secret=_SECRET, kind="github")
    try:
        filler = b'{"kind": "filler"}'
        filler_headers = _github_headers(filler)
        with requests.Session() as session:
            for _ in range(_RATE_LIMIT_EVENTS):
                resp = session.post(
                    f"{base}/hook/rl2", data=filler, headers=filler_headers, timeout=10
                )
                assert resp.status_code == 200
            body = b'{"kind": "important"}'
            headers = _github_headers(body, delivery="retry-me-1")
            resp = session.post(f"{base}/hook/rl2", data=body, headers=headers, timeout=10)
            assert resp.status_code == 429

            with backend._state_lock:
                window = backend._accept_times["rl2"]
                backend._accept_times["rl2"] = deque(
                    t - _RATE_LIMIT_WINDOW_SECONDS - 1.0 for t in window
                )

            resp = session.post(f"{base}/hook/rl2", data=body, headers=headers, timeout=10)
            assert resp.status_code == 200
            resp = session.post(f"{base}/hook/rl2", data=body, headers=headers, timeout=10)
            assert resp.status_code == 200
        messages, _ = backend.poll_messages("rl2", "0", limit=100)
        assert sum(1 for m in messages if "important" in m["text"]) == 1
    finally:
        backend.disconnect()


def test_generic_nonfinite_timestamp_rejected_401() -> None:
    """'nan'/'inf' timestamps are rejected even with a matching signature."""
    backend, base = _start_backend(name="nan", secret=_SECRET, kind="generic")
    try:
        body = b'{"event": "ping"}'
        for ts_str in ("nan", "inf", "-inf", "infinity"):
            digest = hmac.new(
                _SECRET.encode(), f"{ts_str}.".encode() + body, hashlib.sha256
            ).hexdigest()
            headers = {"X-Kiss-Timestamp": ts_str, "X-Kiss-Signature": digest}
            resp = requests.post(f"{base}/hook/nan", data=body, headers=headers, timeout=10)
            assert resp.status_code == 401, ts_str
        assert backend.poll_messages("nan", "0")[0] == []
    finally:
        backend.disconnect()


def test_deliver_only_failure_returns_502_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed delivery -> 502 without idempotency; the retry succeeds -> 202."""
    module_name = _write_delivery_module(tmp_path, monkeypatch, _CONTROLLABLE_DELIVERY_MODULE)
    backend, base = _start_backend(
        name="dl",
        secret=_SECRET,
        kind="github",
        prompt_template="deploy {version}",
        deliver_module=module_name,
        deliver_channel="ops",
    )
    try:
        fail_flag = tmp_path / f"{module_name}.fail"
        out_file = tmp_path / f"{module_name}.out"
        log_file = tmp_path / f"{module_name}.log"
        body = b'{"version": "1.2.3"}'
        headers = _github_headers(body, delivery="deploy-1")

        fail_flag.touch()
        for _ in range(2):
            resp = requests.post(f"{base}/hook/dl", data=body, headers=headers, timeout=10)
            assert resp.status_code == 502
        assert not out_file.exists()

        fail_flag.unlink()
        resp = requests.post(f"{base}/hook/dl", data=body, headers=headers, timeout=10)
        assert resp.status_code == 202
        assert out_file.read_text() == "ops|deploy 1.2.3\n"

        resp = requests.post(f"{base}/hook/dl", data=body, headers=headers, timeout=10)
        assert resp.status_code == 200
        assert out_file.read_text() == "ops|deploy 1.2.3\n"

        log_lines = log_file.read_text().splitlines()
        assert log_lines.count("connect") == 3
        assert log_lines.count("disconnect") == 3
        assert backend.poll_messages("dl", "0")[0] == []
    finally:
        backend.disconnect()


def test_deliver_only_sys_exit_factory_returns_502(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A factory that sys.exit()s (unconfigured module) yields 502, not a hang."""
    module_name = _write_delivery_module(tmp_path, monkeypatch, _SYS_EXIT_DELIVERY_MODULE)
    backend, base = _start_backend(
        name="dx", secret=_SECRET, kind="github", deliver_module=module_name
    )
    try:
        body = b'{"n": 1}'
        resp = requests.post(
            f"{base}/hook/dx", data=body, headers=_github_headers(body), timeout=10
        )
        assert resp.status_code == 502
        body2 = b'{"n": 2}'
        resp = requests.post(
            f"{base}/hook/dx", data=body2, headers=_github_headers(body2), timeout=10
        )
        assert resp.status_code == 502
    finally:
        backend.disconnect()


def test_deliver_only_connect_failure_returns_502_and_disconnects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """connect() returning False fails the delivery; disconnect() still runs."""
    module_name = _write_delivery_module(tmp_path, monkeypatch, _CONNECT_FALSE_DELIVERY_MODULE)
    backend, base = _start_backend(
        name="dc", secret=_SECRET, kind="github", deliver_module=module_name
    )
    try:
        body = b'{"n": 1}'
        resp = requests.post(
            f"{base}/hook/dc", data=body, headers=_github_headers(body), timeout=10
        )
        assert resp.status_code == 502
        assert (tmp_path / f"{module_name}.log").read_text() == "disconnect\n"
    finally:
        backend.disconnect()


def _raw_post_status_latin1(port: int, path: str, headers: dict[str, str]) -> int:
    """POST over a raw socket with latin-1 header bytes (e.g. a real ``²``)."""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        lines = [f"POST {path} HTTP/1.1", f"Host: 127.0.0.1:{port}", "Connection: close"]
        lines.extend(f"{k}: {v}" for k, v in headers.items())
        sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
        data = b""
        while b"\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
        return int(data.split(b" ", 2)[1])


def test_unicode_digit_and_overlong_content_length_rejected() -> None:
    """A latin-1 ``²`` or a 5000-digit Content-Length gets 400, not a crash."""
    backend, base = _start_backend(name="cl2", secret=_SECRET, kind="github")
    try:
        port = backend._bound_port
        assert _raw_post_status_latin1(port, "/hook/cl2", {"Content-Length": "\u00b2"}) == 400
        assert _raw_post_status(port, "/hook/cl2", {"Content-Length": "9" * 5000}) == 400
        body = json.dumps({"event": "ok"}).encode()
        resp = requests.post(
            f"{base}/hook/cl2", data=body, headers=_github_headers(body), timeout=10
        )
        assert resp.status_code == 200
    finally:
        backend.disconnect()


def test_concurrent_same_delivery_id_delivers_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent retries with one delivery id trigger exactly one delivery."""
    module_name = _write_delivery_module(tmp_path, monkeypatch, _CONTROLLABLE_DELIVERY_MODULE)
    backend, base = _start_backend(
        name="conc",
        secret=_SECRET,
        kind="github",
        deliver_module=module_name,
        deliver_channel="chan",
    )
    try:
        body = json.dumps({"event": "x"}).encode()
        headers = _github_headers(body, delivery="dup-conc-1")
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(
                    requests.post, f"{base}/hook/conc", data=body, headers=headers, timeout=10
                )
                for _ in range(8)
            ]
            statuses = sorted(f.result().status_code for f in futures)
        out_file = tmp_path / f"{module_name}.out"
        deliveries = out_file.read_text().splitlines() if out_file.exists() else []
        assert len(deliveries) == 1, deliveries
        assert statuses.count(202) == 1
        assert statuses.count(200) == 7
    finally:
        backend.disconnect()
