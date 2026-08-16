# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the OpenAI-compatible API channel agent — no mocks.

Every HTTP test starts the REAL embedded server on an ephemeral port and
speaks real HTTP with ``requests``.  ``KISS_HOME`` is pointed at a fresh
temp dir per test (and ``KISS_SORCAR_SOCK`` is cleared), so config state
never touches the user's real ``~/.kiss`` and ``kiss.server.sorcar.run``
deterministically fails fast (connection refused on a nonexistent daemon
socket) — exercising the honest 502 no-daemon path.
"""

from __future__ import annotations

import json
import os
import socket
import stat
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import requests

from kiss.agents.third_party_agents import openai_compat_agent as oai_mod
from kiss.agents.third_party_agents.openai_compat_agent import (
    _MAX_BODY_BYTES,
    OpenAICompatAgent,
    OpenAICompatChannelBackend,
    _chat_map_path,
    _conversation_key,
    _lookup_chat_id,
    _store_chat_id,
    _system_prompt_text,
)

_API_KEY = "test-secret-key"


class _EnvSwap:
    """Point ``KISS_HOME`` at a temp dir and clear ``KISS_SORCAR_SOCK``."""

    def __init__(self, target: Path) -> None:
        self._saved_home = os.environ.get("KISS_HOME")
        self._saved_sock = os.environ.get("KISS_SORCAR_SOCK")
        os.environ["KISS_HOME"] = str(target)
        os.environ.pop("KISS_SORCAR_SOCK", None)

    def restore(self) -> None:
        """Restore the original environment values."""
        if self._saved_home is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = self._saved_home
        if self._saved_sock is not None:
            os.environ["KISS_SORCAR_SOCK"] = self._saved_sock


@pytest.fixture()
def isolated_home(tmp_path: Path) -> Iterator[Path]:
    """Per-test KISS_HOME isolation."""
    home = tmp_path / "kiss_home"
    swap = _EnvSwap(home)
    try:
        yield home
    finally:
        swap.restore()


def _auth_tools(agent: OpenAICompatAgent) -> dict[str, Callable[..., str]]:
    """Return the agent's auth trio keyed by function name."""
    return {fn.__name__: fn for fn in agent._get_auth_tools()}


@pytest.fixture()
def api_server(isolated_home: Path) -> Iterator[tuple[str, OpenAICompatChannelBackend]]:
    """Authenticate on an ephemeral port and start the real API server."""
    agent = OpenAICompatAgent()
    result = _auth_tools(agent)["authenticate_openai_compat"](_API_KEY, port="0")
    assert json.loads(result)["ok"] is True
    backend = OpenAICompatChannelBackend()
    assert backend.connect() is True
    assert backend._server is not None
    port = backend._server.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}", backend
    finally:
        backend.disconnect()


def test_agent_instantiation_and_unauthenticated_tools(isolated_home: Path) -> None:
    """Fresh agent: correct name, unauthenticated, only the auth trio."""
    agent = OpenAICompatAgent()
    assert agent.name == "OpenAI-compatible API Agent"
    assert agent._is_authenticated() is False
    names = sorted(fn.__name__ for fn in agent._get_tools())
    assert names == [
        "authenticate_openai_compat",
        "check_openai_compat_auth",
        "clear_openai_compat_auth",
    ]
    assert "Not configured" in _auth_tools(agent)["check_openai_compat_auth"]()


def test_authenticate_persists_config_and_clear(isolated_home: Path) -> None:
    """authenticate persists config with 0600 mode; clear removes it."""
    agent = OpenAICompatAgent()
    tools = _auth_tools(agent)
    result = tools["authenticate_openai_compat"](
        _API_KEY, port="18099", bind_host="127.0.0.1", model_name="some-model"
    )
    assert json.loads(result)["ok"] is True

    cfg_path = isolated_home / "third_party_agents" / "openai_compat" / "config.json"
    assert cfg_path.exists()
    if sys.platform != "win32":
        assert stat.S_IMODE(cfg_path.stat().st_mode) == 0o600
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg == {
        "api_key": _API_KEY,
        "port": "18099",
        "bind_host": "127.0.0.1",
        "model_name": "some-model",
    }

    check = json.loads(tools["check_openai_compat_auth"]())
    assert check["ok"] is True
    assert check["port"] == 18099
    assert check["model_name"] == "some-model"

    assert agent._is_authenticated() is True
    tool_names = [fn.__name__ for fn in agent._get_tools()]
    assert "openai_compat_status" in tool_names

    assert "cleared" in tools["clear_openai_compat_auth"]()
    assert not cfg_path.exists()
    assert agent._is_authenticated() is False


def test_authenticate_rejects_bad_input(isolated_home: Path) -> None:
    """Empty api_key and invalid ports are rejected without persisting."""
    tools = _auth_tools(OpenAICompatAgent())
    assert tools["authenticate_openai_compat"]("") == "api_key cannot be empty."
    assert "Invalid port" in tools["authenticate_openai_compat"](_API_KEY, port="abc")
    assert "Invalid port" in tools["authenticate_openai_compat"](_API_KEY, port="70000")
    assert not (isolated_home / "third_party_agents" / "openai_compat" / "config.json").exists()


def test_get_tools_module_function(isolated_home: Path) -> None:
    """The module-level get_tools() returns a non-empty tool list."""
    tools = oai_mod.get_tools()
    assert tools
    assert "authenticate_openai_compat" in [fn.__name__ for fn in tools]


def test_models_endpoint_requires_no_auth(
    api_server: tuple[str, OpenAICompatChannelBackend],
) -> None:
    """GET /v1/models returns the model list without a bearer token."""
    url, _ = api_server
    resp = requests.get(f"{url}/v1/models", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert data["data"] == [{"id": "kiss-sorcar", "object": "model", "owned_by": "kiss"}]


def test_unknown_paths_return_404(
    api_server: tuple[str, OpenAICompatChannelBackend],
) -> None:
    """Unknown GET and POST paths return OpenAI-style 404 errors."""
    url, _ = api_server
    for resp in (
        requests.get(f"{url}/v1/nope", timeout=10),
        requests.post(f"{url}/v1/nope", json={}, timeout=10),
    ):
        assert resp.status_code == 404
        assert resp.json()["error"]["type"] == "invalid_request_error"


def test_chat_completions_requires_bearer_token(
    api_server: tuple[str, OpenAICompatChannelBackend],
) -> None:
    """POST without or with a wrong bearer token returns 401."""
    url, _ = api_server
    body = {"messages": [{"role": "user", "content": "hi"}]}
    no_auth = requests.post(f"{url}/v1/chat/completions", json=body, timeout=10)
    assert no_auth.status_code == 401
    assert no_auth.json()["error"]["type"] == "invalid_request_error"

    wrong = requests.post(
        f"{url}/v1/chat/completions",
        json=body,
        headers={"Authorization": "Bearer wrong-key"},
        timeout=10,
    )
    assert wrong.status_code == 401
    assert "error" in wrong.json()


def test_malformed_json_returns_400(
    api_server: tuple[str, OpenAICompatChannelBackend],
) -> None:
    """A non-JSON body returns 400 with an OpenAI-style error."""
    url, _ = api_server
    resp = requests.post(
        f"{url}/v1/chat/completions",
        data=b"{not json",
        headers={"Authorization": f"Bearer {_API_KEY}"},
        timeout=10,
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_missing_or_empty_messages_returns_400(
    api_server: tuple[str, OpenAICompatChannelBackend],
) -> None:
    """Missing, empty, user-less, or blank-user messages return 400."""
    url, _ = api_server
    headers = {"Authorization": f"Bearer {_API_KEY}"}
    bad_bodies = [
        {},
        {"messages": []},
        {"messages": "nope"},
        {"messages": [{"role": "system", "content": "no user here"}]},
        {"messages": [{"role": "user", "content": "   "}]},
    ]
    for body in bad_bodies:
        resp = requests.post(f"{url}/v1/chat/completions", json=body, headers=headers, timeout=10)
        assert resp.status_code == 400, body
        assert resp.json()["error"]["type"] == "invalid_request_error"


def test_no_daemon_returns_502_fast(
    api_server: tuple[str, OpenAICompatChannelBackend],
) -> None:
    """With no kiss-web daemon, a valid request gets a fast 502."""
    url, _ = api_server
    start = time.monotonic()
    resp = requests.post(
        f"{url}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers={"Authorization": f"Bearer {_API_KEY}"},
        timeout=30,
    )
    elapsed = time.monotonic() - start
    assert resp.status_code == 502
    error = resp.json()["error"]
    assert error["type"] == "api_error"
    assert "daemon" in error["message"]
    assert elapsed < 10.0
    # No chat mapping is stored for a failed run.
    assert not _chat_map_path().exists()


def test_stream_request_no_daemon_returns_502(
    api_server: tuple[str, OpenAICompatChannelBackend],
) -> None:
    """stream:true still gets the JSON 502 when the daemon is unreachable."""
    url, _ = api_server
    resp = requests.post(
        f"{url}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
        headers={"Authorization": f"Bearer {_API_KEY}"},
        timeout=30,
    )
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "api_error"


def test_conversation_key_is_deterministic_and_canonical(isolated_home: Path) -> None:
    """Keys are stable, ignore metadata fields, and flatten content parts."""
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    assert _conversation_key(messages) == _conversation_key(messages)
    assert _conversation_key(messages) != _conversation_key(messages[:1])
    assert _conversation_key([]) == _conversation_key([])

    with_meta = [
        {"role": "user", "content": "hello", "name": "alice"},
        {"role": "assistant", "content": "hi there", "extra": 1},
    ]
    assert _conversation_key(with_meta) == _conversation_key(messages)

    split_content = [{"type": "text", "text": "hel"}, {"type": "text", "text": "lo"}]
    parts = [
        {"role": "user", "content": split_content},
        {"role": "assistant", "content": "hi there"},
    ]
    assert _conversation_key(parts) == _conversation_key(messages)


def test_chat_map_store_and_lookup_persist(isolated_home: Path) -> None:
    """_store_chat_id persists into chat_map.json under KISS_HOME."""
    key1 = _conversation_key([{"role": "user", "content": "q1"}])
    key2 = _conversation_key([{"role": "user", "content": "q2"}])
    assert _lookup_chat_id(key1) == ""

    _store_chat_id(key1, "chat-abc")
    _store_chat_id(key2, "chat-def")
    assert _lookup_chat_id(key1) == "chat-abc"
    assert _lookup_chat_id(key2) == "chat-def"

    map_path = isolated_home / "third_party_agents" / "openai_compat" / "chat_map.json"
    assert map_path == _chat_map_path()
    assert map_path.exists()
    if sys.platform != "win32":
        assert stat.S_IMODE(map_path.stat().st_mode) == 0o600
    assert json.loads(map_path.read_text(encoding="utf-8"))[key1] == "chat-abc"

    # Overwriting a key keeps a single up-to-date entry.
    _store_chat_id(key1, "chat-xyz")
    assert _lookup_chat_id(key1) == "chat-xyz"


def test_followup_request_maps_to_same_chat(isolated_home: Path) -> None:
    """A client's next request (with the assistant reply appended) maps back.

    Mirrors exactly what the handler stores after a successful run: the
    daemon chat id under the full request message list AND under that
    list plus the assistant reply, so the lookup key of the follow-up
    request (all messages except its last user message) hits.
    """
    first_request = [{"role": "user", "content": "What is 2+2?"}]
    assistant_reply = "4"
    _store_chat_id(_conversation_key(first_request), "chat-123")
    with_reply = first_request + [{"role": "assistant", "content": assistant_reply}]
    _store_chat_id(_conversation_key(with_reply), "chat-123")

    followup = with_reply + [{"role": "user", "content": "And 3+3?"}]
    lookup_key = _conversation_key(followup[:-1])
    assert _lookup_chat_id(lookup_key) == "chat-123"


def test_poll_messages_and_send_message(isolated_home: Path) -> None:
    """poll_messages returns ([], oldest); send_message raises."""
    backend = OpenAICompatChannelBackend()
    assert backend.poll_messages("any", "42") == ([], "42")
    with pytest.raises(RuntimeError, match="inbound-only"):
        backend.send_message("any", "text")


def test_openai_compat_status_tool(
    api_server: tuple[str, OpenAICompatChannelBackend],
) -> None:
    """openai_compat_status reports bound=True while serving, False after."""
    url, backend = api_server
    running = json.loads(backend.openai_compat_status())
    assert running["ok"] is True
    assert running["bound"] is True
    assert f"127.0.0.1:{running['port']}" in url

    # A backend that is NOT running the server probes the configured
    # port (0 in this isolated config) and reports it unbound.
    other = OpenAICompatChannelBackend()
    probed = json.loads(other.openai_compat_status())
    assert probed["ok"] is True
    assert probed["bound"] is False


def test_status_tool_unconfigured(isolated_home: Path) -> None:
    """openai_compat_status without config reports not configured."""
    status = json.loads(OpenAICompatChannelBackend().openai_compat_status())
    assert status["ok"] is False
    assert "Not configured" in status["error"]


def test_connect_fails_without_config(isolated_home: Path) -> None:
    """connect() returns False with a helpful message when unconfigured."""
    backend = OpenAICompatChannelBackend()
    assert backend.connect() is False
    assert "No OpenAI-compatible API config" in backend.connection_info


def test_system_prompt_text_collects_system_and_developer_messages() -> None:
    """System/developer messages become the system_prompt, in order.

    The handler passes ``_system_prompt_text(messages)`` as the
    ``system_prompt=`` argument of ``kiss.server.sorcar.run``; this
    exercises that collection directly.
    """
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
        {"role": "developer", "content": [{"type": "text", "text": "Use JSON."}]},
        {"role": "assistant", "content": "ok"},
        "not-a-dict",
        {"role": "system", "content": "   "},
        {"role": "system", "content": "Second rule."},
    ]
    assert _system_prompt_text(messages) == "Be brief.\n\nUse JSON.\n\nSecond rule."
    assert _system_prompt_text([{"role": "user", "content": "hi"}]) == ""
    assert _system_prompt_text([]) == ""


def test_request_with_system_message_no_daemon_returns_502(
    api_server: tuple[str, OpenAICompatChannelBackend],
) -> None:
    """A request carrying a system message reaches the daemon call.

    With no daemon it still gets the OpenAI-style 502 (i.e. the
    system message is accepted and forwarded, not a crash or a 400).
    """
    url, _ = api_server
    resp = requests.post(
        f"{url}/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "hello"},
            ]
        },
        headers={"Authorization": f"Bearer {_API_KEY}"},
        timeout=30,
    )
    assert resp.status_code == 502
    error = resp.json()["error"]
    assert error["type"] == "api_error"
    assert "daemon" in error["message"]
    assert error["param"] is None
    assert error["code"] is None
    # No chat mapping is stored for a failed run.
    assert not _chat_map_path().exists()


def _raw_post_chat(port: int, extra_headers: list[str]) -> tuple[int, dict]:
    """POST /v1/chat/completions with hand-built headers over a raw socket.

    Args:
        port: The API server's TCP port on 127.0.0.1.
        extra_headers: Extra raw header lines (e.g. a hostile
            ``Content-Length``); no body is sent.

    Returns:
        Tuple of (HTTP status code, decoded JSON response body).
    """
    lines = [
        "POST /v1/chat/completions HTTP/1.1",
        "Host: 127.0.0.1",
        f"Authorization: Bearer {_API_KEY}",
        "Connection: close",
        *extra_headers,
    ]
    request = ("\r\n".join(lines) + "\r\n\r\n").encode("ascii")
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.settimeout(10)
        sock.sendall(request)
        raw = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            raw += chunk
    head, _, payload = raw.partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    return status, json.loads(payload)


def test_bad_content_length_is_rejected_without_hanging(
    api_server: tuple[str, OpenAICompatChannelBackend],
) -> None:
    """Negative, non-decimal, missing, and oversized Content-Length.

    Each gets an immediate OpenAI-style 400 (or 413 for oversized)
    without the server trying to read a body — a negative value used
    to reach ``rfile.read(-1)``, which blocks until EOF.
    """
    _, backend = api_server
    assert backend._server is not None
    port = backend._server.server_address[1]
    cases: list[tuple[list[str], int]] = [
        (["Content-Length: -1"], 400),
        (["Content-Length: nope"], 400),
        ([], 400),  # missing Content-Length entirely
        ([f"Content-Length: {_MAX_BODY_BYTES + 1}"], 413),
    ]
    for extra_headers, expected_status in cases:
        start = time.monotonic()
        status, body = _raw_post_chat(port, extra_headers)
        assert time.monotonic() - start < 10.0, extra_headers
        assert status == expected_status, extra_headers
        assert body["error"]["type"] == "invalid_request_error"


def test_store_chat_id_is_atomic_and_survives_corrupt_map(isolated_home: Path) -> None:
    """A pre-existing corrupt chat_map.json is replaced atomically.

    The corrupt map reads as empty, the store rewrites it via a temp
    file + ``os.replace`` (0600, no leftover temp file), and the
    result reloads as valid JSON.
    """
    map_path = _chat_map_path()
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text("{corrupt json", encoding="utf-8")

    key = _conversation_key([{"role": "user", "content": "q"}])
    assert _lookup_chat_id(key) == ""  # corrupt map treated as empty

    _store_chat_id(key, "chat-1")
    assert _lookup_chat_id(key) == "chat-1"
    assert json.loads(map_path.read_text(encoding="utf-8")) == {key: "chat-1"}
    if sys.platform != "win32":
        assert stat.S_IMODE(map_path.stat().st_mode) == 0o600
    assert not map_path.with_name(map_path.name + ".tmp").exists()
