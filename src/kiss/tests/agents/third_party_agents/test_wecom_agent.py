# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the WeCom channel agent — no mocks or test doubles.

Runs a real local HTTP server as the WeCom group-robot webhook
receiver to assert outbound payload shapes and error handling, and
verifies the auth-trio config persistence.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler
from typing import Any

import pytest

import kiss.agents.third_party_agents.wecom_agent as wecom_mod
from kiss.agents.third_party_agents._backend_utils import ThreadedHTTPServer
from kiss.agents.third_party_agents.wecom_agent import (
    WeComAgent,
    WeComChannelBackend,
    get_tools,
)

_AUTH_TRIO = {"check_wecom_auth", "authenticate_wecom", "clear_wecom_auth"}


class _WebhookReceiver:
    """Real local HTTP server standing in for WeCom's webhook endpoint.

    Records every request body and answers with a configurable JSON
    body (``errcode: 0`` by default).
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.response_body: dict[str, Any] = {"errcode": 0, "errmsg": "ok"}
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                receiver.requests.append({"path": self.path, "json": json.loads(body)})
                payload = json.dumps(receiver.response_body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:  # type: ignore[override]
                pass

        self.server = ThreadedHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/cgi-bin/webhook/send?key=testkey"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Shut the receiver down."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5.0)


@pytest.fixture()
def receiver() -> Any:
    """A running local webhook receiver, stopped after the test."""
    rec = _WebhookReceiver()
    yield rec
    rec.stop()


@pytest.fixture(autouse=True)
def _clean_config() -> Any:
    """Start and finish every test with no persisted WeCom config."""
    wecom_mod._config.clear()
    yield
    wecom_mod._config.clear()


def test_agent_unauthenticated_exposes_only_auth_trio() -> None:
    """Unauthenticated agents expose exactly the auth tool trio."""
    agent = WeComAgent()
    assert agent.name == "WeCom Agent"
    assert agent._is_authenticated() is False
    tool_names = {t.__name__ for t in agent._get_tools()}
    assert tool_names == _AUTH_TRIO
    check = next(t for t in agent._get_tools() if t.__name__ == "check_wecom_auth")
    assert "authenticate_wecom" in check()


def test_authenticate_persists_and_clear_removes() -> None:
    """authenticate_wecom persists 0600 config; clear removes it."""
    agent = WeComAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    result = tools["authenticate_wecom"]("https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=k")
    assert json.loads(result)["ok"] is True

    path = wecom_mod._config.path
    assert path.exists()
    if sys.platform != "win32":
        assert (path.stat().st_mode & 0o777) == 0o600
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == {"webhook_url": "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=k"}

    assert agent._is_authenticated() is True
    tool_names = {t.__name__ for t in agent._get_tools()}
    assert {"post_message", "post_markdown"} <= tool_names
    assert json.loads(tools["check_wecom_auth"]())["ok"] is True

    assert "cleared" in tools["clear_wecom_auth"]().lower()
    assert not path.exists()
    assert agent._is_authenticated() is False
    assert {t.__name__ for t in agent._get_tools()} == _AUTH_TRIO


def test_authenticate_rejects_empty_url() -> None:
    """An empty webhook URL is rejected without persisting config."""
    agent = WeComAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    assert "empty" in tools["authenticate_wecom"]("   ")
    assert not wecom_mod._config.path.exists()


def test_fresh_agent_loads_persisted_config() -> None:
    """A new agent instance picks up the persisted config."""
    agent = WeComAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    tools["authenticate_wecom"]("https://x?key=1")
    fresh = WeComAgent()
    assert fresh._is_authenticated() is True
    assert fresh._backend._webhook_url == "https://x?key=1"


def test_get_tools_module_function() -> None:
    """The module-level get_tools() returns a non-empty tool list."""
    assert len(get_tools()) >= 3


def test_post_message_shape(receiver: _WebhookReceiver) -> None:
    """post_message sends the WeCom text payload with mentions."""
    agent = WeComAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    tools["authenticate_wecom"](receiver.url)

    result = json.loads(agent._backend.post_message("hello 团队", mentioned_list="alice, @all"))
    assert result["ok"] is True
    assert receiver.requests[-1]["json"] == {
        "msgtype": "text",
        "text": {"content": "hello 团队", "mentioned_list": ["alice", "@all"]},
    }


def test_post_markdown_shape(receiver: _WebhookReceiver) -> None:
    """post_markdown sends the WeCom markdown payload."""
    agent = WeComAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    tools["authenticate_wecom"](receiver.url)

    assert json.loads(agent._backend.post_markdown("# heading\n> quote"))["ok"] is True
    assert receiver.requests[-1]["json"] == {
        "msgtype": "markdown",
        "markdown": {"content": "# heading\n> quote"},
    }


def test_send_message_posts_text(receiver: _WebhookReceiver) -> None:
    """send_message posts a plain text payload, ignoring channel/thread."""
    agent = WeComAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    tools["authenticate_wecom"](receiver.url)

    agent._backend.send_message("ignored-channel", "plain text", thread_ts="ignored")
    assert receiver.requests[-1]["json"] == {"msgtype": "text", "text": {"content": "plain text"}}


def test_errcode_nonzero_raises_and_tools_report(receiver: _WebhookReceiver) -> None:
    """A non-zero errcode raises from send_message and yields ok:false tools."""
    agent = WeComAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    tools["authenticate_wecom"](receiver.url)
    receiver.response_body = {"errcode": 93000, "errmsg": "invalid webhook url"}

    with pytest.raises(RuntimeError, match="93000"):
        agent._backend.send_message("", "boom")
    assert json.loads(agent._backend.post_message("boom"))["ok"] is False
    assert json.loads(agent._backend.post_markdown("boom"))["ok"] is False


def test_poll_messages_returns_empty() -> None:
    """poll_messages always reports no messages (outbound-only adapter)."""
    backend = WeComChannelBackend()
    assert backend.poll_messages("any", "42") == ([], "42")


def test_connect_reflects_config_state() -> None:
    """connect() fails without config and succeeds after authenticate."""
    backend = WeComChannelBackend()
    assert backend.connect() is False
    assert "No WeCom config" in backend.connection_info

    agent = WeComAgent()
    tools = {t.__name__: t for t in agent._get_tools()}
    tools["authenticate_wecom"]("https://x?key=1")
    assert backend.connect() is True
    assert backend.connection_info == "WeCom configured"
