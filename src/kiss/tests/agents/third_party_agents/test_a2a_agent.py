# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the A2A channel agent — no mocks or test doubles.

Exercises the real embedded JSON-RPC server over HTTP: agent card
discovery, bearer-token enforcement, message normalization and queue
draining, task lifecycle (submitted -> completed via ``send_message``),
the anti-ping-pong turn cap, malformed/oversized request rejection, the
audit log, and the full outbound tool loop (``a2a_discover`` /
``a2a_call`` / ``a2a_get_task``) pointed at the embedded server.
"""

from __future__ import annotations

import json
import socket
import sys
import uuid
from typing import Any

import pytest
import requests

import kiss.agents.third_party_agents.a2a_agent as a2a_mod
from kiss.agents.third_party_agents.a2a_agent import (
    A2AAgent,
    A2AChannelBackend,
    _config,
)

_AUTH_TRIO = {"check_a2a_auth", "authenticate_a2a", "clear_a2a_auth"}
_BACKEND_TOOLS = {"a2a_discover", "a2a_call", "a2a_get_task"}


@pytest.fixture(autouse=True)
def _clean_a2a_config():
    """Start and end each test with no persisted A2A config."""
    _config.clear()
    yield
    _config.clear()


def _free_port() -> int:
    """Reserve and return a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _tool_names(tools: list) -> set[str]:
    """Return the ``__name__`` set of a tool list."""
    return {t.__name__ for t in tools}


def _auth_tool(agent: A2AAgent, name: str):
    """Return the named auth tool closure from an agent."""
    return next(t for t in agent._get_auth_tools() if t.__name__ == name)


def _configured_backend(port: int, token: str = "") -> A2AChannelBackend:
    """Authenticate the channel and return a connected poll-mode backend."""
    agent = A2AAgent()
    result = _auth_tool(agent, "authenticate_a2a")(
        bind_host="127.0.0.1", port=str(port), token=token
    )
    assert json.loads(result)["ok"] is True
    backend = a2a_mod._make_backend()
    assert backend.connect() is True
    return backend


def _rpc(base: str, method: str, params: dict, token: str = "") -> requests.Response:
    """POST one JSON-RPC request to the embedded server."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method, "params": params}
    return requests.post(base + "/", json=payload, headers=headers, timeout=10)


def _send_params(text: str, context_id: str = "") -> dict:
    """Build ``message/send`` params for *text*."""
    message = {
        "role": "user",
        "parts": [{"kind": "text", "text": text}],
        "messageId": str(uuid.uuid4()),
        "kind": "message",
    }
    if context_id:
        message["contextId"] = context_id
    return {"message": message}


def test_agent_instantiation_unauthenticated() -> None:
    """A fresh agent is unauthenticated and exposes only the auth trio."""
    agent = A2AAgent()
    assert agent.name == "A2A Agent"
    assert agent._is_authenticated() is False
    assert _tool_names(agent._get_tools()) == _AUTH_TRIO


def test_authenticate_persists_check_and_clear() -> None:
    """The auth trio persists, reports, and clears config on disk."""
    agent = A2AAgent()
    assert "Not configured" in _auth_tool(agent, "check_a2a_auth")()

    result = _auth_tool(agent, "authenticate_a2a")(
        bind_host="127.0.0.1", port="18099", token="tok", agent_name="Test Peer"
    )
    assert json.loads(result)["ok"] is True
    assert _config.path.exists()
    assert "third_party_agents/a2a" in str(_config.path)
    if sys.platform != "win32":
        assert _config.path.stat().st_mode & 0o777 == 0o600
    saved = json.loads(_config.path.read_text(encoding="utf-8"))
    assert saved == {
        "bind_host": "127.0.0.1",
        "port": "18099",
        "token": "tok",
        "agent_name": "Test Peer",
    }

    status = json.loads(_auth_tool(agent, "check_a2a_auth")())
    assert status["ok"] is True
    assert status["port"] == "18099"
    assert status["token_set"] is True
    assert status["agent_name"] == "Test Peer"

    assert agent._is_authenticated() is True
    tools = _tool_names(agent._get_tools())
    assert _AUTH_TRIO <= tools
    assert _BACKEND_TOOLS <= tools

    assert "cleared" in _auth_tool(agent, "clear_a2a_auth")()
    assert not _config.path.exists()
    assert agent._is_authenticated() is False


def test_authenticate_rejects_bad_input() -> None:
    """Empty host/port and non-integer ports are rejected."""
    agent = A2AAgent()
    assert "cannot be empty" in _auth_tool(agent, "authenticate_a2a")(bind_host="", port="1")
    assert "must be an integer" in _auth_tool(agent, "authenticate_a2a")(port="http")
    assert not _config.path.exists()


def test_get_tools_module_function() -> None:
    """The module-level ``get_tools()`` returns a non-empty tool list."""
    tools = a2a_mod.get_tools()
    assert tools
    assert _AUTH_TRIO <= _tool_names(tools)


def test_inbound_outbound_end_to_end() -> None:
    """Full lifecycle over real HTTP: card, auth, send, poll, reply, get."""
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    backend = _configured_backend(port, token="sekret")
    try:
        # Agent card is served on both well-known paths.
        for path in ("/.well-known/agent-card.json", "/.well-known/agent.json"):
            card = requests.get(base + path, timeout=10).json()
            assert card["name"] == "KISS Sorcar"
            assert card["protocolVersion"] == "0.2"
            assert card["capabilities"] == {"streaming": False}
            assert card["skills"][0]["id"] == "general"
        assert requests.get(base + "/nope", timeout=10).status_code == 404

        # Missing/wrong bearer token -> 401, and nothing is queued.
        assert _rpc(base, "message/send", _send_params("hi")).status_code == 401
        assert _rpc(base, "message/send", _send_params("hi"), token="wrong").status_code == 401
        assert backend.poll_messages("", "0") == ([], "0")

        # message/send with the token submits a task and queues the text.
        resp = _rpc(base, "message/send", _send_params("hello agent", "ctx-1"), token="sekret")
        assert resp.status_code == 200
        task = resp.json()["result"]
        assert task["kind"] == "task"
        assert task["contextId"] == "ctx-1"
        assert task["status"] == {"state": "submitted"}

        messages, _ = backend.poll_messages("", "0")
        assert len(messages) == 1
        msg = messages[0]
        assert msg["user"] == "a2a-peer"
        assert msg["text"] == "hello agent"
        assert msg["channel_id"] == "ctx-1"
        assert msg["thread_ts"] == task["id"]
        assert float(msg["ts"]) > 0

        # tasks/get before the reply: still submitted, no artifacts.
        got = _rpc(base, "tasks/get", {"id": task["id"]}, token="sekret").json()["result"]
        assert got["status"] == {"state": "submitted"}
        assert "artifacts" not in got

        # send_message records the reply; tasks/get now shows it completed.
        backend.send_message("ctx-1", "here is my answer", task["id"])
        got = _rpc(base, "tasks/get", {"id": task["id"]}, token="sekret").json()["result"]
        assert got["status"] == {"state": "completed"}
        assert got["artifacts"][0]["parts"] == [{"kind": "text", "text": "here is my answer"}]
        assert got["artifacts"][0]["artifactId"]

        # Error paths: unknown task, unknown method, malformed/oversized bodies.
        err = _rpc(base, "tasks/get", {"id": "missing"}, token="sekret").json()["error"]
        assert err["code"] == -32001
        err = _rpc(base, "bogus/method", {}, token="sekret").json()["error"]
        assert err["code"] == -32601
        bad = requests.post(
            base + "/",
            data=b"not json",
            headers={"Authorization": "Bearer sekret"},
            timeout=10,
        )
        assert bad.json()["error"]["code"] == -32700
        with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
            sock.sendall(
                b"POST / HTTP/1.1\r\nHost: peer\r\nAuthorization: Bearer sekret\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: 2097152\r\n\r\n"
            )
            status_line = sock.recv(4096).decode("utf-8", "replace").splitlines()[0]
        assert " 413 " in status_line

        # Replying to a context with no pending task raises, with either
        # an unknown task id or an empty thread_ts.
        with pytest.raises(RuntimeError):
            backend.send_message("no-such-ctx", "reply", "no-such-task")
        with pytest.raises(RuntimeError):
            backend.send_message("no-such-ctx", "reply", "")
        # Completing an already-completed task raises too.
        with pytest.raises(RuntimeError):
            backend.send_message("ctx-1", "again", task["id"])

        # Audit log has one JSONL line per inbound request.
        audit = _config.path.parent / "a2a_audit.jsonl"
        assert audit.exists()
        entries = [json.loads(line) for line in audit.read_text().splitlines()]
        assert all({"ts", "method", "contextId", "ok"} <= set(e) for e in entries)
        assert {"method": "message/send", "contextId": "ctx-1", "ok": True} in [
            {k: e[k] for k in ("method", "contextId", "ok")} for e in entries
        ]

        # Full outbound loop using the backend's own tools against the server.
        peer = A2AChannelBackend()
        card_result = json.loads(peer.a2a_discover(base))
        assert card_result["ok"] is True
        assert card_result["card"]["name"] == "KISS Sorcar"

        denied = json.loads(peer.a2a_call(base, "hi"))
        assert denied == {"ok": False, "error": "HTTP 401"}

        call = json.loads(peer.a2a_call(base, "ping from peer", "ctx-out", token="sekret"))
        assert call["ok"] is True
        out_task = call["result"]
        assert out_task["contextId"] == "ctx-out"

        messages, _ = backend.poll_messages("ctx-out", "0")
        assert [m["text"] for m in messages] == ["ping from peer"]
        assert messages[0]["thread_ts"] == out_task["id"]

        backend.send_message("ctx-out", "pong", out_task["id"])
        got = json.loads(peer.a2a_get_task(base, out_task["id"], token="sekret"))
        assert got["ok"] is True
        assert got["result"]["status"] == {"state": "completed"}
        assert got["result"]["artifacts"][0]["parts"][0]["text"] == "pong"

        # Outbound tools never raise on unreachable peers.
        dead = f"http://127.0.0.1:{_free_port()}"
        assert json.loads(peer.a2a_discover(dead))["ok"] is False
        assert json.loads(peer.a2a_call(dead, "hi"))["ok"] is False
        assert json.loads(peer.a2a_get_task(dead, "t1"))["ok"] is False
    finally:
        backend.disconnect()


def test_message_send_without_context_creates_one() -> None:
    """message/send without contextId gets a fresh uuid context."""
    port = _free_port()
    backend = _configured_backend(port)
    try:
        resp = _rpc(f"http://127.0.0.1:{port}", "message/send", _send_params("no ctx"))
        task = resp.json()["result"]
        assert uuid.UUID(task["contextId"])
        messages, _ = backend.poll_messages("", "0")
        assert messages[0]["channel_id"] == task["contextId"]
    finally:
        backend.disconnect()


def test_multi_part_text_concatenation() -> None:
    """All text parts are concatenated; non-text parts are ignored."""
    port = _free_port()
    backend = _configured_backend(port)
    try:
        params = _send_params("first", "ctx-parts")
        params["message"]["parts"] = [
            {"kind": "text", "text": "first"},
            {"kind": "file", "uri": "http://x/y"},
            {"kind": "text", "text": "second"},
        ]
        resp = _rpc(f"http://127.0.0.1:{port}", "message/send", params)
        assert resp.json()["result"]["status"] == {"state": "submitted"}
        messages, _ = backend.poll_messages("ctx-parts", "0")
        assert messages[0]["text"] == "first\nsecond"
    finally:
        backend.disconnect()


def test_turn_limit_per_context_per_hour() -> None:
    """The 21st inbound message in one context within an hour is rejected."""
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    backend = _configured_backend(port)
    try:
        for i in range(20):
            resp = _rpc(base, "message/send", _send_params(f"msg {i}", "ctx-limit"))
            assert "result" in resp.json(), f"message {i} unexpectedly rejected"
        err = _rpc(base, "message/send", _send_params("msg 20", "ctx-limit")).json()["error"]
        assert err == {"code": -32000, "message": "turn limit exceeded"}
        # Other contexts are unaffected.
        assert "result" in _rpc(base, "message/send", _send_params("hi", "ctx-other")).json()
        # The rejected message was neither queued nor given a task.
        messages, _ = backend.poll_messages("ctx-limit", "0", limit=50)
        assert len(messages) == 20
    finally:
        backend.disconnect()


def _raw_post(port: int, headers: str, payload: bytes = b"") -> str:
    """Send one raw HTTP POST and return the response status line."""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(b"POST / HTTP/1.1\r\nHost: peer\r\n" + headers.encode("latin-1") + b"\r\n\r\n")
        if payload:
            sock.sendall(payload)
        return sock.recv(4096).decode("utf-8", "replace").splitlines()[0]


def _audit_entries() -> list[dict]:
    """Read all JSONL entries from the audit log."""
    audit = _config.path.parent / "a2a_audit.jsonl"
    return [json.loads(line) for line in audit.read_text().splitlines()]


def test_same_context_pending_tasks_each_get_their_own_reply() -> None:
    """Two pending tasks in one context are completed independently."""
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    backend = _configured_backend(port)
    try:
        task_a = _rpc(base, "message/send", _send_params("question A", "ctx")).json()["result"]
        task_b = _rpc(base, "message/send", _send_params("question B", "ctx")).json()["result"]
        assert task_a["id"] != task_b["id"]

        messages, _ = backend.poll_messages("ctx", "0")
        by_text = {m["text"]: m for m in messages}
        assert by_text["question A"]["thread_ts"] == task_a["id"]
        assert by_text["question B"]["thread_ts"] == task_b["id"]

        # Reply to A first even though B is the newest pending task: the
        # reply must land on A, not on the latest pending task.
        backend.send_message("ctx", "answer A", by_text["question A"]["thread_ts"])
        got_a = _rpc(base, "tasks/get", {"id": task_a["id"]}).json()["result"]
        got_b = _rpc(base, "tasks/get", {"id": task_b["id"]}).json()["result"]
        assert got_a["status"] == {"state": "completed"}
        assert got_a["artifacts"][0]["parts"][0]["text"] == "answer A"
        assert got_b["status"] == {"state": "submitted"}

        backend.send_message("ctx", "answer B", by_text["question B"]["thread_ts"])
        got_b = _rpc(base, "tasks/get", {"id": task_b["id"]}).json()["result"]
        assert got_b["artifacts"][0]["parts"][0]["text"] == "answer B"

        # A reply carrying a task id from a different context raises.
        _rpc(base, "message/send", _send_params("other", "ctx2"))
        messages, _ = backend.poll_messages("ctx2", "0")
        with pytest.raises(RuntimeError):
            backend.send_message("ctx", "cross", messages[0]["thread_ts"])
    finally:
        backend.disconnect()


def test_empty_thread_ts_falls_back_to_newest_pending() -> None:
    """With an empty thread_ts, the newest pending task completes."""
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    backend = _configured_backend(port)
    try:
        _rpc(base, "message/send", _send_params("old", "ctx"))
        task_new = _rpc(base, "message/send", _send_params("new", "ctx")).json()["result"]
        backend.send_message("ctx", "reply", "")
        got = _rpc(base, "tasks/get", {"id": task_new["id"]}).json()["result"]
        assert got["status"] == {"state": "completed"}
    finally:
        backend.disconnect()


def test_invalid_envelope_params_and_message_are_rejected() -> None:
    """Bad envelopes and params get JSON-RPC errors, never a crash."""
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    backend = _configured_backend(port)
    try:
        # params.message is not an object -> -32602, audited, server alive.
        resp = requests.post(
            base + "/",
            json={"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {"message": "x"}},
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["error"]["code"] == -32602
        audited = [e for e in _audit_entries() if e["method"] == "message/send" and not e["ok"]]
        assert audited

        # params is not an object -> -32602.
        resp = requests.post(
            base + "/",
            json={"jsonrpc": "2.0", "id": 2, "method": "message/send", "params": "bad"},
            timeout=10,
        )
        assert resp.json()["error"]["code"] == -32602

        # Missing/wrong jsonrpc version -> -32600.
        resp = requests.post(base + "/", json={"id": 3, "method": "tasks/get"}, timeout=10)
        assert resp.json()["error"]["code"] == -32600
        bad_version = {"jsonrpc": "1.0", "id": 4, "method": "x"}
        resp = requests.post(base + "/", json=bad_version, timeout=10)
        assert resp.json()["error"]["code"] == -32600

        # The handler is still alive: a good request round-trips.
        good = _rpc(base, "message/send", _send_params("still alive", "ctx-ok"))
        assert good.json()["result"]["status"] == {"state": "submitted"}
    finally:
        backend.disconnect()


def test_bad_content_length_is_handled_and_server_stays_up() -> None:
    """Negative, non-numeric, missing, and oversized Content-Length."""
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    backend = _configured_backend(port)
    try:
        assert " 400 " in _raw_post(port, "Content-Length: -1")
        assert " 400 " in _raw_post(port, "Content-Length: nope")
        assert " 400 " in _raw_post(port, "Content-Length: \u00b2")
        assert " 400 " in _raw_post(port, "Content-Type: application/json")
        assert " 413 " in _raw_post(port, f"Content-Length: {2 * 1024 * 1024}")
        # The server survives all of the above.
        good = _rpc(base, "message/send", _send_params("after bad lengths", "ctx-len"))
        assert good.json()["result"]["status"] == {"state": "submitted"}
    finally:
        backend.disconnect()


def test_non_loopback_bind_requires_token() -> None:
    """A non-loopback bind_host with an empty token is rejected everywhere."""
    agent = A2AAgent()
    auth = _auth_tool(agent, "authenticate_a2a")
    for host in ("0.0.0.0", "192.168.1.5", "example.com"):
        assert "without a token" in auth(bind_host=host, port="18099", token="")
    assert not _config.path.exists()
    assert "between 0 and 65535" in auth(port="65536")

    # Loopback hosts and tokened non-loopback binds are accepted.
    assert json.loads(auth(bind_host="localhost", port="18099", token=""))["ok"] is True
    assert json.loads(auth(bind_host="::1", port="18099", token=""))["ok"] is True
    assert json.loads(auth(bind_host="0.0.0.0", port="18099", token="tok"))["ok"] is True

    # connect() independently refuses an unsafe persisted config.
    _config.save({"bind_host": "0.0.0.0", "port": str(_free_port()), "token": ""})
    backend = a2a_mod._make_backend()
    assert backend.connect() is False
    assert "without a token" in backend.connection_info


def test_connect_fails_without_config_or_bad_port() -> None:
    """connect() is False when unconfigured or the port cannot be bound."""
    backend = A2AChannelBackend()
    assert backend.connect() is False
    assert "No A2A config" in backend.connection_info

    port = _free_port()
    first = _configured_backend(port)
    try:
        second = a2a_mod._make_backend()
        assert second.connect() is False
        assert "bind failed" in second.connection_info
    finally:
        first.disconnect()


def test_overlong_content_length_rejected_400() -> None:
    """A 5000-digit Content-Length gets 400 instead of an int() crash."""
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    backend = _configured_backend(port)
    try:
        assert " 400 " in _raw_post(port, "Content-Length: " + "9" * 5000)
        good = _rpc(base, "message/send", _send_params("after overlong length", "ctx-len2"))
        assert good.json()["result"]["status"] == {"state": "submitted"}
    finally:
        backend.disconnect()


def test_invalid_jsonrpc_id_types_rejected_32600() -> None:
    """JSON-RPC ids must be a string, number, or null; others get -32600."""
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    backend = _configured_backend(port)
    try:
        for bad_id in ({"invalid": "object-id"}, ["array-id"], True):
            payload: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": bad_id,
                "method": "message/send",
                "params": _send_params("bad id", "ctx-id"),
            }
            resp = requests.post(base + "/", json=payload, timeout=10)
            body = resp.json()
            assert body["error"]["code"] == -32600, body
            assert "id" in body["error"]["message"]
        # Valid ids still work: string, number, and null.
        for good_id in ("str-id", 7, None):
            payload = {
                "jsonrpc": "2.0",
                "id": good_id,
                "method": "message/send",
                "params": _send_params("good id", "ctx-id2"),
            }
            resp = requests.post(base + "/", json=payload, timeout=10)
            body = resp.json()
            assert body.get("id") == good_id
            assert body["result"]["status"] == {"state": "submitted"}
    finally:
        backend.disconnect()
