# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the Telegram agent's legacy (non-Muse) transport.

The legacy path used to import ``python-telegram-bot``'s sync ``Bot``,
which no longer exists in any version that runs on Python 3.14 and is
not a project dependency.  These tests drive the whole legacy flow
(``authenticate_telegram`` -> config file -> fresh ``TelegramAgent`` ->
``connect``/``check_telegram_auth``/``_make_backend``) against a local
HTTP emulator of the Bot API, with ``KISS_HOME`` pointed at a temp dir.

Not covered here: the Muse branch of ``_TelegramBot`` (``session is
None``), which needs the Muse daemon; that daemon requires
``socket.SO_PEERCRED`` (Linux) and cannot run on macOS, so the branch
is unreachable in this environment without a test double.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from kiss.agents.third_party_agents.telegram_agent import (
    TelegramAgent,
    TelegramChannelBackend,
    _config,
    _make_backend,
    _TelegramBot,
)

VALID_TOKEN = "123456:VALID-TOKEN"
SEEN_REQUESTS: list[dict[str, Any]] = []


class _BotApiHandler(BaseHTTPRequestHandler):
    """Minimal Telegram Bot API emulator recording every request."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence the default stderr access log."""

    def do_POST(self) -> None:  # noqa: N802
        """Answer ``/bot<token>/<method>`` like api.telegram.org would."""
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        _, _, rest = self.path.partition("/bot")
        token, _, method = rest.partition("/")
        SEEN_REQUESTS.append(
            {
                "token": token,
                "method": method,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": body,
            }
        )
        if token != VALID_TOKEN:
            self._reply(401, {"ok": False, "description": "Unauthorized"})
        elif method == "getMe":
            self._reply(
                200,
                {
                    "ok": True,
                    "result": {"id": 123456, "username": "emu_bot", "first_name": "Emu"},
                },
            )
        elif method in ("sendMessage", "sendPhoto"):
            self._reply(200, {"ok": True, "result": {"message_id": 77}})
        elif method == "badJson":
            self._reply_raw(200, b"<html>not json</html>")
        elif method == "errorObject":
            self._reply(500, {"error": {"message": "boom"}})
        elif method == "listBody":
            self._reply_raw(200, b"[]")
        else:
            self._reply(404, {"ok": False, "description": "Not Found"})

    def _reply(self, status: int, payload: dict[str, Any]) -> None:
        self._reply_raw(status, json.dumps(payload).encode())

    def _reply_raw(self, status: int, raw: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def api_base() -> Any:
    """Run the Bot API emulator for one test and yield its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _BotApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    SEEN_REQUESTS.clear()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def legacy_env(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Isolate config under a temp ``KISS_HOME`` and force legacy auth."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    monkeypatch.setenv("KISS_MUSE_AUTH", "0")
    return tmp_path


def _auth_tools(agent: TelegramAgent) -> dict[str, Any]:
    return {tool.__name__: tool for tool in agent._get_auth_tools()}


def test_authenticate_then_check_and_persist(legacy_env: Any, api_base: str) -> None:
    """A valid token is validated over HTTP, saved with mode 0600 and reusable."""
    agent = TelegramAgent()
    agent._backend._api_base = api_base
    tools = _auth_tools(agent)
    assert tools["check_telegram_auth"]().startswith("Not authenticated with Telegram")

    result = json.loads(tools["authenticate_telegram"](f"  {VALID_TOKEN}  "))
    assert result == {
        "ok": True,
        "message": "Telegram token saved and validated.",
        "username": "emu_bot",
        "id": 123456,
    }
    assert json.loads(tools["check_telegram_auth"]()) == {
        "ok": True,
        "username": "emu_bot",
        "first_name": "Emu",
        "id": 123456,
    }

    config_path = legacy_env / "third_party_agents" / "telegram" / "config.json"
    assert _config.path == config_path
    assert json.loads(config_path.read_text()) == {"bot_token": VALID_TOKEN}
    assert (config_path.stat().st_mode & 0o777) == 0o600

    # Legacy mode puts the real token in the URL and sends no bearer header.
    get_me = SEEN_REQUESTS[0]
    assert get_me["token"] == VALID_TOKEN
    assert "authorization" not in get_me["headers"]

    # A fresh agent (what a kiss-web restart does) picks the token up from disk.
    fresh = TelegramAgent()
    assert fresh._is_authenticated()
    assert isinstance(fresh._backend._bot, _TelegramBot)
    fresh._backend._api_base = api_base
    assert fresh._backend.connect() is True
    assert fresh._backend._connection_info == "Authenticated as @emu_bot"
    assert fresh._backend._bot_token() == VALID_TOKEN
    assert fresh._backend._request_headers() == {}

    backend = _make_backend()
    assert isinstance(backend._bot, _TelegramBot)
    assert backend._bot.token == VALID_TOKEN

    assert tools["clear_telegram_auth"]() == "Telegram authentication cleared."
    assert not config_path.exists()
    assert not TelegramAgent()._is_authenticated()


def test_authenticate_rejects_bad_or_empty_token(legacy_env: Any, api_base: str) -> None:
    """A rejected token is reported as an error and nothing is written."""
    agent = TelegramAgent()
    agent._backend._api_base = api_base
    tools = _auth_tools(agent)
    assert tools["authenticate_telegram"]("   ") == "bot_token cannot be empty."

    result = json.loads(tools["authenticate_telegram"]("999:WRONG"))
    assert result["ok"] is False
    assert "getMe failed: HTTP 401 Unauthorized" in result["error"]
    assert not _config.path.exists()
    assert not agent._is_authenticated()


def test_connect_reports_failure_and_missing_config(legacy_env: Any, api_base: str) -> None:
    """``connect`` fails closed for a bad stored token and for no config."""
    backend = TelegramChannelBackend()
    backend._api_base = api_base
    assert backend.connect() is False
    assert backend._connection_info == "No Telegram token found."

    _config.save({"bot_token": "999:WRONG"})
    assert backend.connect() is False
    assert backend._connection_info.startswith("Telegram auth failed: Telegram API getMe failed")


def test_make_backend_exits_without_config(legacy_env: Any, capsys: pytest.CaptureFixture) -> None:
    """Poll-mode startup exits with instructions when no token is stored."""
    with pytest.raises(SystemExit):
        _make_backend()
    assert "Not authenticated. Run: kiss-telegram -t 'authenticate'" in capsys.readouterr().out


def test_bot_adapter_multipart_and_error_envelopes(legacy_env: Any, api_base: str) -> None:
    """Uploads go multipart; malformed and error envelopes raise RuntimeError."""
    import io

    import requests

    backend = TelegramChannelBackend()
    backend._api_base = api_base
    bot = _TelegramBot(backend, VALID_TOKEN, requests.Session())

    sent = bot.send_message(42, "hi", reply_to_message_id=5)
    assert sent.message_id == 77
    assert json.loads(SEEN_REQUESTS[-1]["body"]) == {
        "chat_id": 42,
        "text": "hi",
        "reply_to_message_id": 5,
    }

    photo = bot.send_photo(42, io.BytesIO(b"\x89PNG"), caption="cap")
    assert photo.message_id == 77
    assert SEEN_REQUESTS[-1]["headers"]["content-type"].startswith("multipart/form-data")

    with pytest.raises(RuntimeError, match="badJson failed: HTTP 200 $"):
        bot._call("badJson")
    with pytest.raises(RuntimeError, match="errorObject failed: HTTP 500 boom"):
        bot._call("errorObject")
    with pytest.raises(RuntimeError, match="listBody failed: HTTP 200 $"):
        bot._call("listBody")
