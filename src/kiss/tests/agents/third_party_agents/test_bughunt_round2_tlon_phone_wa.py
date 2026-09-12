# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Round-2 integration tests for tlon_agent and phone_control_agent.

Runs real in-process HTTP servers that record request method/path/body and
return service-shaped JSON. No mocks, patches, or test doubles.

Bugs covered:
- Tlon ``poke`` must follow the Urbit Eyre protocol: PUT to ``/~/channel/{uid}``
  with a JSON array of action objects, an incrementing message id, and the
  configured ship name (RuntimeError when the ship is not configured).
- Tlon ``post_message`` memo author must be the configured ship, not ``"~"``.
- phone_control ``poll_messages`` must only return SMS from the requested
  ``channel_id`` sender (empty channel_id returns all).

(The former WhatsApp webhook-verification tests were removed: the WhatsApp
agent now uses the QR-paired whatsapp-mcp bridge and has no Meta webhook.)
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

from kiss.agents.third_party_agents.phone_control_agent import PhoneControlChannelBackend
from kiss.agents.third_party_agents.phone_control_agent import _config as _phone_config
from kiss.agents.third_party_agents.tlon_agent import TlonChannelBackend
from kiss.agents.third_party_agents.tlon_agent import _config as _tlon_config


class _RecordingHandler(BaseHTTPRequestHandler):
    """Handler that records requests and returns service-shaped JSON."""

    requests_seen: list[dict[str, Any]] = []

    def _record(self, body: bytes) -> None:
        self.requests_seen.append(
            {"method": self.command, "path": self.path, "body": body.decode() if body else ""}
        )

    def _respond_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def do_GET(self) -> None:  # noqa: N802
        """Serve phone REST and Graph API GET endpoints."""
        self._record(b"")
        parsed = urlparse(self.path)
        if parsed.path == "/api/device/info":
            self._respond_json(200, {"device_name": "testphone"})
        elif parsed.path == "/api/sms/messages":
            self._respond_json(
                200,
                {
                    "messages": [
                        {"timestamp": 1, "from": "SENDER_A", "body": "from A", "id": "1"},
                        {"timestamp": 2, "from": "SENDER_B", "body": "from B", "id": "2"},
                    ]
                },
            )
        else:
            self._respond_json(
                200, {"verified_name": "Test Biz", "display_phone_number": "+1555"}
            )

    def do_POST(self) -> None:  # noqa: N802
        """Serve Eyre login and legacy poke endpoints."""
        self._record(self._read_body())
        if self.path == "/~/login" or self.path.startswith("/~/channel"):
            self.send_response(204)
            self.end_headers()
        else:
            self._respond_json(200, {})

    def do_PUT(self) -> None:  # noqa: N802
        """Serve Eyre channel PUT (poke) endpoints."""
        self._record(self._read_body())
        if self.path.startswith("/~/channel"):
            self.send_response(204)
            self.end_headers()
        else:
            self._respond_json(404, {"error": "Not Found"})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence request logging."""


def _free_port() -> int:
    """Return an ephemeral free TCP port on localhost."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_server() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _channel_requests() -> list[dict[str, Any]]:
    return [
        r for r in _RecordingHandler.requests_seen if r["path"].startswith("/~/channel")
    ]


class _ConfigBackup:
    """Back up and restore a ChannelConfig JSON file around a test."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._backup: str | None = None

    def save(self) -> None:
        """Record the current config contents (if any)."""
        self._backup = self._path.read_text() if self._path.exists() else None

    def restore(self) -> None:
        """Restore the original config contents (or remove the file)."""
        if self._backup is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(self._backup)
        elif self._path.exists():
            self._path.unlink()


class TestTlonPokeProtocol:
    """Tlon poke must follow the Eyre channel protocol with the configured ship."""

    def setup_method(self) -> None:
        _RecordingHandler.requests_seen = []
        self._server, self.base = _start_server()
        self._cfg = _ConfigBackup(_tlon_config.path)
        self._cfg.save()
        _tlon_config.save(
            {"ship_url": self.base, "code": "lidlut-tabwed", "ship": "~sampel-palnet"}
        )
        self.backend = TlonChannelBackend()
        assert self.backend.connect() is True

    def teardown_method(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._cfg.restore()

    def test_poke_uses_eyre_put_with_uid_array_and_ship(self) -> None:
        """poke must PUT a one-element JSON array to /~/channel/{uid}."""
        result = json.loads(self.backend.poke("channels", "channel-action", '{"a": 1}'))
        assert result["ok"] is True
        pokes = _channel_requests()
        assert pokes, f"no /~/channel request seen: {_RecordingHandler.requests_seen}"
        req = pokes[-1]
        assert req["method"] == "PUT", f"poke must use PUT, saw {req['method']}"
        assert req["path"].startswith("/~/channel/"), req["path"]
        uid = req["path"][len("/~/channel/") :]
        assert uid, f"poke PUT path has no channel uid segment: {req['path']}"
        body = json.loads(req["body"])
        assert isinstance(body, list) and len(body) == 1, f"body must be a JSON array: {body}"
        action = body[0]
        assert action["action"] == "poke"
        assert action["ship"] == "sampel-palnet", f"ship wrong/missing: {action}"
        assert action["app"] == "channels"
        assert action["mark"] == "channel-action"
        assert action["json"] == {"a": 1}
        assert isinstance(action["id"], int)

    def test_poke_ids_increment_per_connection(self) -> None:
        """Consecutive pokes must carry strictly increasing message ids."""
        self.backend.poke("channels", "channel-action", '{"a": 1}')
        self.backend.poke("channels", "channel-action", '{"a": 2}')
        pokes = _channel_requests()
        assert len(pokes) == 2, pokes
        id1 = json.loads(pokes[0]["body"])[0]["id"]
        id2 = json.loads(pokes[1]["body"])[0]["id"]
        assert id2 > id1, f"poke ids must increase: {id1} then {id2}"
        uid1 = pokes[0]["path"]
        uid2 = pokes[1]["path"]
        assert uid1 == uid2, f"pokes on one connection must reuse the channel uid: {uid1} {uid2}"

    def test_post_message_author_is_configured_ship(self) -> None:
        """post_message memo author must be the configured ship, not '~'."""
        result = json.loads(self.backend.post_message("~zod/my-group", "chat", "hello"))
        assert result["ok"] is True
        pokes = _channel_requests()
        assert pokes, f"no poke request seen: {_RecordingHandler.requests_seen}"
        body = json.loads(pokes[-1]["body"])
        action = body[0] if isinstance(body, list) else body
        memo = action["json"]["channel-action"]["post"]["action"]["add"]["memo"]
        assert memo["author"] == "~sampel-palnet", f"memo author wrong: {memo}"


class TestTlonPokeWithoutShip:
    """poke must raise RuntimeError when no ship is configured (legacy config)."""

    def setup_method(self) -> None:
        _RecordingHandler.requests_seen = []
        self._server, self.base = _start_server()
        self._cfg = _ConfigBackup(_tlon_config.path)
        self._cfg.save()
        _tlon_config.save({"ship_url": self.base, "code": "lidlut-tabwed"})
        self.backend = TlonChannelBackend()
        assert self.backend.connect() is True, "config without 'ship' must still load"

    def teardown_method(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._cfg.restore()

    def test_poke_without_ship_raises_runtime_error(self) -> None:
        """poke without a configured ship must raise instead of sending junk."""
        with pytest.raises(RuntimeError):
            self.backend.poke("channels", "channel-action", '{"a": 1}')
        assert not _channel_requests(), "no Eyre request must be sent without a ship"


class TestPhoneControlSenderFilter:
    """poll_messages must only return SMS from the requested channel_id sender."""

    def setup_method(self) -> None:
        _RecordingHandler.requests_seen = []
        self._server, self.base = _start_server()
        self._cfg = _ConfigBackup(_phone_config.path)
        self._cfg.save()
        port = self._server.server_address[1]
        _phone_config.save({"device_ip": "127.0.0.1", "device_port": str(port)})
        self.backend = PhoneControlChannelBackend()
        assert self.backend.connect() is True

    def teardown_method(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._cfg.restore()

    def test_poll_filters_to_requested_sender(self) -> None:
        """poll_messages('SENDER_A', '') must not return SENDER_B's SMS."""
        messages, _ = self.backend.poll_messages("SENDER_A", "")
        assert len(messages) == 1, f"expected only SENDER_A messages, got {messages}"
        assert messages[0]["user"] == "SENDER_A"
        assert messages[0]["text"] == "from A"

    def test_poll_with_empty_channel_returns_all(self) -> None:
        """poll_messages('', '') keeps returning every sender's SMS."""
        messages, new_oldest = self.backend.poll_messages("", "")
        assert len(messages) == 2, messages
        assert new_oldest == "2"
