# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for Google Chat sign-in: Composio user auth and Chat bot keys.

The Composio path runs the real googleapiclient Chat service and the
real Composio SDK against a local Composio API emulator
(``composio_test_utils``), whose proxy forwards to a local Chat API
server.  The service-account path loads a real, freshly generated RSA
service-account key.  No mocks or patches.
"""

from __future__ import annotations

import json
import stat
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, cast

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from kiss.agents.third_party_agents.googlechat.googlechat_sea import (
    _SERVICE,
    GoogleChatAgent,
    _load_service,
    _service_account_path,
)
from kiss.tests.agents.third_party_agents.composio_test_utils import (
    TOKEN,
    fake_composio,
    reset_state,
)
from kiss.tests.agents.third_party_agents.recording_http import RecordingServer, recording_server
from kiss.tests.conftest import IS_WINDOWS


class _ChatHandler(BaseHTTPRequestHandler):
    """Serves spaces.list for the Composio-injected token, else 401."""

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """Answer GET /v1/spaces."""
        cast(RecordingServer, self.server).requests.append(
            {"path": self.path, "authorization": self.headers.get("Authorization")}
        )
        ok = self.headers.get("Authorization") == f"Bearer {TOKEN}"
        body = json.dumps(
            {"spaces": [{"name": "spaces/A", "displayName": "Team", "type": "ROOM"}]}
            if ok
            else {"error": {"code": 401, "message": "Invalid Credentials"}}
        ).encode()
        self.send_response(200 if ok else 401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:  # type: ignore[override]
        """Silence request logging."""


@pytest.fixture(autouse=True)
def _fresh_state():
    """Start and end every test without a connection or service account key."""
    reset_state(_SERVICE)
    _service_account_path().unlink(missing_ok=True)
    yield
    reset_state(_SERVICE)
    _service_account_path().unlink(missing_ok=True)


@pytest.fixture()
def chat(monkeypatch):
    """Run a local Chat API behind the local Composio emulator.

    Yields:
        ``(composio, chat_server)``, with chat.googleapis.com rerouted
        to the local Chat server.
    """
    with recording_server(_ChatHandler) as server, fake_composio(monkeypatch) as composio:
        composio.upstream_overrides["https://chat.googleapis.com"] = server.base_url
        yield composio, server


def _write_service_account_key(path: Path) -> None:
    """Write a real service-account JSON key with a fresh RSA private key."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    path.write_text(
        json.dumps(
            {
                "type": "service_account",
                "project_id": "kiss-test",
                "private_key_id": "k1",
                "private_key": pem,
                "client_email": "bot@kiss-test.iam.gserviceaccount.com",
                "client_id": "1",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        )
    )


def _tools(agent: GoogleChatAgent) -> dict[str, Any]:
    return {t.__name__: t for t in agent._get_auth_tools()}


def test_unauthenticated_without_key_or_connection() -> None:
    """With neither a key nor a connection, the agent offers the auth tools."""
    agent = GoogleChatAgent()
    assert agent._is_authenticated() is False
    assert agent._backend._service is None
    assert _load_service() is None
    assert "Not authenticated with Google Chat" in _tools(agent)["check_googlechat_auth"]()
    assert list(_tools(agent)) == [
        "check_googlechat_auth",
        "authenticate_googlechat",
        "clear_googlechat_auth",
        "finish_googlechat_auth",
        "authenticate_googlechat_service_account",
    ]


def test_connect_link_flow_builds_composio_service(chat) -> None:
    """authenticate -> approve -> finish gives a Chat service behind Composio."""
    composio, server = chat
    agent = GoogleChatAgent()
    tools = _tools(agent)
    started = json.loads(tools["authenticate_googlechat"]())
    assert started["status"] == "consent_required"
    composio.accounts[started["verification_uri"].rsplit("/", 1)[1]] = "ACTIVE"
    assert json.loads(tools["finish_googlechat_auth"]())["ok"] is True
    assert agent._is_authenticated() is True
    result = json.loads(agent._backend.list_spaces())
    assert result["spaces"] == [{"name": "spaces/A", "display_name": "Team", "type": "ROOM"}]
    assert server.requests[-1]["authorization"] == f"Bearer {TOKEN}"
    # A new agent picks the recorded connection up on its own.
    assert GoogleChatAgent()._backend._service is not None


def test_service_account_tool_copies_key_and_authenticates(tmp_path: Path) -> None:
    """A key elsewhere is loaded, copied to the default path at 0600, and used."""
    key_file = tmp_path / "bot.json"
    _write_service_account_key(key_file)
    agent = GoogleChatAgent()
    result = json.loads(_tools(agent)["authenticate_googlechat_service_account"](str(key_file)))
    assert result == {"ok": True, "message": "Google Chat service account loaded."}
    assert agent._backend._service is not None
    assert agent._is_authenticated() is True
    saved = _service_account_path()
    assert json.loads(saved.read_text()) == json.loads(key_file.read_text())
    assert IS_WINDOWS or stat.S_IMODE(saved.stat().st_mode) == 0o600
    # The saved default key now loads without a path.
    again = json.loads(_tools(GoogleChatAgent())["authenticate_googlechat_service_account"]())
    assert again["ok"] is True
    assert _load_service() is not None


def test_clear_forgets_the_service_account_key(chat, tmp_path: Path) -> None:
    """clear_googlechat_auth deletes the stored key and drops the loaded service."""
    composio, _server = chat
    key_file = tmp_path / "bot.json"
    _write_service_account_key(key_file)
    agent = GoogleChatAgent()
    tools = _tools(agent)
    assert json.loads(tools["authenticate_googlechat_service_account"](str(key_file)))["ok"]
    assert _service_account_path().exists() and agent._backend._service is not None
    assert json.loads(tools["check_googlechat_auth"]())["ok"] is True

    assert tools["clear_googlechat_auth"]() == "Google Chat connection cleared."
    assert not _service_account_path().exists()
    assert agent._backend._service is None
    assert agent._is_authenticated() is False
    assert "Not authenticated with Google Chat" in tools["check_googlechat_auth"]()
    # Nothing was connected through Composio, so nothing was deleted there.
    assert composio.deleted == []
    # A fresh agent no longer finds a key either.
    assert GoogleChatAgent()._backend._service is None


def test_malformed_stored_key_is_not_authenticated() -> None:
    """An unreadable default service_account.json leaves the agent unauthenticated."""
    path = _service_account_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    agent = GoogleChatAgent()
    assert agent._backend._service is None
    assert agent._is_authenticated() is False
    assert _load_service() is None
    assert "Not authenticated with Google Chat" in _tools(agent)["check_googlechat_auth"]()

    # Valid JSON that is not a service-account key fails the same way.
    path.write_text(json.dumps({"type": "service_account", "client_email": "x"}))
    assert GoogleChatAgent()._is_authenticated() is False
    # ... and so does a key file that cannot be read at all.
    path.write_text(json.dumps({"type": "service_account"}))
    if not IS_WINDOWS:
        path.chmod(0)
        try:
            assert GoogleChatAgent()._is_authenticated() is False
        finally:
            path.chmod(0o600)


def test_service_account_tool_rejects_missing_or_bad_keys(tmp_path: Path) -> None:
    """Missing or malformed keys fail clearly and are not copied."""
    tools = _tools(GoogleChatAgent())
    missing = json.loads(tools["authenticate_googlechat_service_account"]())
    assert missing["ok"] is False and "service_account.json" in missing["error"]
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    broken = json.loads(tools["authenticate_googlechat_service_account"](str(bad)))
    assert broken["ok"] is False and str(bad) in broken["error"]
    assert not _service_account_path().exists()
