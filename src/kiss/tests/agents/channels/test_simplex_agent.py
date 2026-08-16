# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the SimpleX Chat channel agent.

Runs a real ``websockets.sync.server`` speaking the simplex-chat CLI
WebSocket protocol (corrId echo, canned command responses, one
unsolicited ``newChatItems`` push on connect) — no mocks or test
doubles anywhere.
"""

from __future__ import annotations

import json
import stat
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from websockets.sync.server import Server, ServerConnection, serve

import kiss.agents.third_party_agents.simplex_agent as simplex_mod
from kiss.agents.third_party_agents.simplex_agent import (
    SimpleXAgent,
    SimpleXChannelBackend,
    _config,
    get_tools,
)

_NEW_CHAT_ITEMS_EVENT: dict[str, Any] = {
    "resp": {
        "type": "newChatItems",
        "chatItems": [
            {
                "chatInfo": {
                    "type": "direct",
                    "contact": {"localDisplayName": "alice"},
                },
                "chatItem": {
                    "chatDir": {"type": "directRcv"},
                    "meta": {"itemId": 42, "itemTs": "2026-01-02T03:04:05Z"},
                    "content": {
                        "type": "rcvMsgContent",
                        "msgContent": {"type": "text", "text": "hello bot"},
                    },
                },
            },
            {
                "chatInfo": {
                    "type": "group",
                    "groupInfo": {"localDisplayName": "team"},
                },
                "chatItem": {
                    "chatDir": {
                        "type": "groupRcv",
                        "groupMember": {"localDisplayName": "bob"},
                    },
                    "meta": {"itemId": 43, "itemTs": "2026-01-02T03:04:06Z"},
                    "content": {
                        "type": "rcvMsgContent",
                        "msgContent": {"type": "text", "text": "hi from group"},
                    },
                },
            },
            {
                "chatInfo": {
                    "type": "direct",
                    "contact": {"localDisplayName": "alice"},
                },
                "chatItem": {
                    "chatDir": {"type": "directSnd"},
                    "meta": {"itemId": 44, "itemTs": "2026-01-02T03:04:07Z"},
                    "content": {
                        "type": "sndMsgContent",
                        "msgContent": {"type": "text", "text": "own sent msg"},
                    },
                },
            },
        ],
    }
}

_CONTACTS_RESP: dict[str, Any] = {
    "type": "contactsList",
    "contacts": [
        {"localDisplayName": "alice"},
        {"localDisplayName": "bob"},
    ],
}

_SEND_OK_RESP: dict[str, Any] = {"type": "newChatItems", "chatItems": []}

_CMD_ERROR_RESP: dict[str, Any] = {
    "type": "chatCmdError",
    "chatError": {"type": "error", "errorType": {"type": "contactNotFound"}},
}

_SHOW_ADDRESS_RESP: dict[str, Any] = {
    "type": "userContactLink",
    "contactLink": {"connReqContact": "simplex:/contact#existing-address"},
}


def _handler(conn: ServerConnection) -> None:
    """Speak the simplex-chat CLI WS protocol for one client connection."""
    conn.send(json.dumps(_NEW_CHAT_ITEMS_EVENT))
    for raw in conn:
        frame = json.loads(raw)
        corr_id = frame["corrId"]
        cmd = frame["cmd"]
        if cmd == "/contacts":
            resp: dict[str, Any] = _CONTACTS_RESP
        elif cmd.startswith("@'alice'"):
            resp = _SEND_OK_RESP
        elif cmd == "/address":
            resp = _CMD_ERROR_RESP
        elif cmd == "/show_address":
            resp = _SHOW_ADDRESS_RESP
        else:
            resp = _CMD_ERROR_RESP
        conn.send(json.dumps({"corrId": corr_id, "resp": resp}))


@pytest.fixture(autouse=True)
def _isolated_kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point KISS_HOME at a per-test temp dir so ~/.kiss is never touched."""
    home = tmp_path / "kiss_home"
    monkeypatch.setenv("KISS_HOME", str(home))
    return home


@pytest.fixture
def simplex_server() -> Iterator[str]:
    """Start a real sync WebSocket server on a free port; yield its ws URL."""
    server: Server = serve(_handler, "127.0.0.1", 0)
    port = server.socket.getsockname()[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"ws://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5.0)


def _authenticate(agent: SimpleXAgent, ws_url: str) -> dict[str, Callable[..., str]]:
    """Run the authenticate_simplex tool and return the tool map."""
    tools = {t.__name__: t for t in agent._get_tools()}
    result = json.loads(tools["authenticate_simplex"](ws_url=ws_url))
    assert result["ok"] is True
    return {t.__name__: t for t in agent._get_tools()}


def test_agent_unauthenticated_state() -> None:
    """Fresh agent: correct name, unauthenticated, only the auth trio."""
    agent = SimpleXAgent()
    assert agent.name == "SimpleX Agent"
    assert agent._is_authenticated() is False
    names = sorted(t.__name__ for t in agent._get_tools())
    assert names == [
        "authenticate_simplex",
        "check_simplex_auth",
        "clear_simplex_auth",
    ]
    tools = {t.__name__: t for t in agent._get_tools()}
    msg = tools["check_simplex_auth"]()
    assert "Not configured" in msg
    assert "authenticate_simplex" in msg


def test_auth_persistence_and_clear(_isolated_kiss_home: Path) -> None:
    """authenticate_simplex persists config (0600); check reports it; clear removes it."""
    agent = SimpleXAgent()
    tools = _authenticate(agent, "ws://127.0.0.1:5225")

    config_path = _config.path
    assert config_path == (_isolated_kiss_home / "third_party_agents" / "simplex" / "config.json")
    assert config_path.exists()
    if sys.platform != "win32":
        assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    assert json.loads(config_path.read_text())["ws_url"] == "ws://127.0.0.1:5225"

    assert agent._is_authenticated() is True
    check = json.loads(tools["check_simplex_auth"]())
    assert check == {"ok": True, "ws_url": "ws://127.0.0.1:5225"}

    # Backend tool methods appear after authentication.
    names = {t.__name__ for t in agent._get_tools()}
    assert {
        "send_simplex_message",
        "list_simplex_contacts",
        "get_simplex_address",
    } <= names

    # A fresh agent picks up the persisted config.
    assert SimpleXAgent()._is_authenticated() is True

    result = tools["clear_simplex_auth"]()
    assert "cleared" in result.lower()
    assert not config_path.exists()
    assert agent._is_authenticated() is False


def test_authenticate_rejects_empty_url() -> None:
    """authenticate_simplex('') returns an error and stores nothing."""
    agent = SimpleXAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    assert tools["authenticate_simplex"](ws_url="  ") == "ws_url cannot be empty."
    assert agent._is_authenticated() is False
    assert not _config.path.exists()


def test_get_tools_module_function() -> None:
    """get_tools() returns a non-empty list (at least the auth trio)."""
    tools = get_tools()
    assert len(tools) >= 3
    assert simplex_mod.get_tools.__doc__


def test_end_to_end_over_real_websocket(simplex_server: str) -> None:
    """Connect, send, list contacts, poll, error handling, disconnect — for real."""
    agent = SimpleXAgent()
    tools = _authenticate(agent, simplex_server)
    backend = agent._backend

    assert backend.connect() is True
    assert simplex_server in backend.connection_info

    # Send via the LLM tool to a known contact.
    sent = json.loads(tools["send_simplex_message"](contact="alice", text="hi alice"))
    assert sent == {"ok": True}

    # List contacts through the real /contacts round-trip.
    contacts = json.loads(tools["list_simplex_contacts"]())
    assert contacts["ok"] is True
    assert contacts["contacts"] == ["alice", "bob"]

    # The unsolicited newChatItems event pushed on connect was queued while
    # waiting for command responses; poll drains the normalized messages.
    messages, cursor = backend.poll_messages("", "0", limit=10)
    assert cursor == "0"
    assert [m["text"] for m in messages] == ["hello bot", "hi from group"]

    direct = messages[0]
    assert direct["user"] == "alice"
    assert direct["username"] == "alice"
    assert direct["channel_id"] == "alice"
    assert direct["thread_ts"] == "42"
    assert direct["ts"] == "2026-01-02T03:04:05Z"
    assert backend.is_from_bot(direct) is False

    group = messages[1]
    assert group["user"] == "bob"
    assert group["channel_id"] == "team"
    assert group["thread_ts"] == "43"

    # Sent-direction items are never queued, and is_from_bot flags them.
    assert backend.is_from_bot({"direction": "directSnd"}) is True
    assert backend.is_from_bot({"direction": "groupSnd"}) is True

    # chatCmdError: send_message raises, the tool reports ok False.
    with pytest.raises(RuntimeError, match="chatCmdError"):
        backend.send_message("nosuch", "boo")
    failed = json.loads(tools["send_simplex_message"](contact="nosuch", text="boo"))
    assert failed["ok"] is False
    assert "chatCmdError" in failed["error"]

    # /address reports the address exists; falls back to /show_address.
    address = json.loads(tools["get_simplex_address"]())
    assert address == {"ok": True, "address": "simplex:/contact#existing-address"}

    backend.disconnect()
    assert backend._ws is None


def test_poll_filters_by_channel(simplex_server: str) -> None:
    """poll_messages(channel_id=...) keeps only that chat's messages."""
    agent = SimpleXAgent()
    _authenticate(agent, simplex_server)
    backend = agent._backend
    assert backend.connect() is True

    messages, _ = backend.poll_messages("team", "0", limit=10)
    assert [m["text"] for m in messages] == ["hi from group"]
    # The non-matching direct message was discarded; the queue is empty now.
    again, _ = backend.poll_messages("", "0", limit=10)
    assert again == []
    backend.disconnect()


def test_tool_errors_when_unreachable() -> None:
    """Tools return ok:false JSON (never raise) when the CLI is unreachable."""
    backend = SimpleXChannelBackend()
    result = json.loads(backend.send_simplex_message("alice", "hi"))
    assert result["ok"] is False
    assert "not configured" in result["error"].lower()

    backend._ws_url = "ws://127.0.0.1:9"  # discard port: connection refused
    for tool in (
        lambda: backend.send_simplex_message("alice", "hi"),
        backend.list_simplex_contacts,
        backend.get_simplex_address,
    ):
        result = json.loads(tool())
        assert result["ok"] is False
        assert result["error"]


def test_connect_without_config() -> None:
    """connect() fails cleanly when no config has been saved."""
    backend = SimpleXChannelBackend()
    assert backend.connect() is False
    assert "No SimpleX Chat config" in backend.connection_info
