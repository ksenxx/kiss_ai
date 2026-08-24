# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the shared HTTP-server START helper (G-R1).

Ten channel modules used to carry a verbatim copy of the embedded
HTTP-server start block (construct ThreadedHTTPServer, start a daemon
``serve_forever`` thread, reset both attributes and log on failure).
The block now lives in ``_backend_utils.start_http_server`` — the START
half matching the existing ``stop_http_server``.

Every test runs real servers on real sockets — no mocks or doubles.
"""

from __future__ import annotations

import logging
import socket
import urllib.request
from http.server import BaseHTTPRequestHandler
from typing import Any

import pytest

from kiss.agents.third_party_agents._backend_utils import (
    start_http_server,
    stop_http_server,
)

logger = logging.getLogger(__name__)


class _PingHandler(BaseHTTPRequestHandler):
    """Minimal handler answering every GET with 200 'pong'."""

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        """Answer 200 with a fixed body."""
        body = b"pong"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - http.server API
        """Silence request logging."""


def _blocked_port() -> tuple[socket.socket, int]:
    """Bind and listen on an ephemeral wildcard port, returning (socket, port)."""
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("", 0))
    blocker.listen(1)
    return blocker, blocker.getsockname()[1]


class TestStartHttpServerHelper:
    """Direct tests of the shared helper."""

    def test_success_serves_requests_and_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The started server answers a real HTTP request; port is logged."""
        with caplog.at_level(logging.INFO, logger=__name__):
            server, thread, error = start_http_server(
                ("127.0.0.1", 0),
                _PingHandler,
                log=logger,
                started_log="test server started on port %d",
                error_prefix="test bind failed",
                error_log="could not start test server: %s",
            )
        try:
            assert error is None
            assert server is not None and thread is not None
            assert thread.daemon and thread.is_alive()
            port = server.server_address[1]
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
                assert resp.read() == b"pong"
            assert f"test server started on port {port}" in caplog.text
        finally:
            stop_http_server(server, thread)

    def test_started_log_none_logs_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``started_log=None`` leaves success logging to the caller."""
        with caplog.at_level(logging.INFO, logger=__name__):
            server, thread, error = start_http_server(
                ("127.0.0.1", 0),
                _PingHandler,
                log=logger,
                started_log=None,
                error_prefix="test bind failed",
                error_log="could not start test server: %s",
            )
        try:
            assert error is None
            assert caplog.text == ""
        finally:
            stop_http_server(server, thread)

    def test_bind_conflict_returns_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An occupied port yields (None, None, error) and a warning log."""
        blocker, port = _blocked_port()
        try:
            with caplog.at_level(logging.WARNING, logger=__name__):
                server, thread, error = start_http_server(
                    ("", port),
                    _PingHandler,
                    log=logger,
                    started_log="never logged %d",
                    error_prefix="test bind failed",
                    error_log="could not start test server: %s",
                )
            assert (server, thread) == (None, None)
            assert error is not None and error.startswith("test bind failed: ")
            assert "could not start test server" in caplog.text
            assert "never logged" not in caplog.text
        finally:
            blocker.close()

    def test_string_port_coerced_and_valueerror_caught_when_listed(self) -> None:
        """A garbage string port fails as a caught ValueError when listed."""
        server, thread, error = start_http_server(
            ("127.0.0.1", "not-a-port"),
            _PingHandler,
            log=logger,
            started_log=None,
            error_prefix="test bind failed",
            error_log="could not start test server: %s",
            catch=(OSError, ValueError),
        )
        assert (server, thread) == (None, None)
        assert error is not None and "not-a-port" in error

    def test_uncaught_exception_type_propagates(self) -> None:
        """Exception types outside *catch* are not swallowed."""
        with pytest.raises(ValueError):
            start_http_server(
                ("127.0.0.1", "not-a-port"),
                _PingHandler,
                log=logger,
                started_log=None,
                error_prefix="test bind failed",
                error_log="could not start test server: %s",
            )


def _port_arg_backends() -> list[tuple[Any, str]]:
    """Backends whose start method takes the port as an argument."""
    from kiss.agents.third_party_agents.dingtalk_agent import DingTalkChannelBackend
    from kiss.agents.third_party_agents.line_agent import LineChannelBackend
    from kiss.agents.third_party_agents.synology_chat_agent import (
        SynologyChatChannelBackend,
    )
    from kiss.agents.third_party_agents.whatsapp_agent import WhatsAppChannelBackend
    from kiss.agents.third_party_agents.zalo_agent import ZaloChannelBackend

    return [
        (WhatsAppChannelBackend, "WhatsApp webhook bind failed: "),
        (ZaloChannelBackend, "Zalo webhook bind failed: "),
        (LineChannelBackend, "LINE webhook bind failed: "),
        (DingTalkChannelBackend, "DingTalk callback bind failed: "),
        (SynologyChatChannelBackend, "Synology webhook bind failed: "),
    ]


class TestPortArgumentBackends:
    """The five backends passing an explicit port to their start method."""

    @pytest.mark.parametrize("backend_cls,prefix", _port_arg_backends())
    def test_start_and_stop(self, backend_cls: Any, prefix: str) -> None:
        """Starting on an ephemeral port succeeds and serves a thread."""
        backend = backend_cls()
        try:
            assert backend._start_webhook_server(0) is True
            assert backend._webhook_server is not None
            assert backend._webhook_server.server_address[1] > 0
            assert backend._webhook_thread is not None
            assert backend._webhook_thread.is_alive()
        finally:
            backend.disconnect()
        assert backend._webhook_server is None
        assert backend._webhook_thread is None

    @pytest.mark.parametrize("backend_cls,prefix", _port_arg_backends())
    def test_bind_conflict(self, backend_cls: Any, prefix: str) -> None:
        """An occupied port fails with the module's exact error prefix."""
        blocker, port = _blocked_port()
        backend = backend_cls()
        try:
            assert backend._start_webhook_server(port) is False
            assert backend._webhook_server is None
            assert backend._webhook_thread is None
            assert backend.connection_info.startswith(prefix)
        finally:
            blocker.close()


class TestAttributePortBackends:
    """The five backends reading the port from configured attributes."""

    def test_weixin_start_and_bad_port(self) -> None:
        """Weixin starts on an ephemeral port; a garbage port is caught."""
        from kiss.agents.third_party_agents.weixin_agent import WeixinChannelBackend

        backend = WeixinChannelBackend()
        backend._port = "0"
        try:
            assert backend._start_callback_server() is True
            assert backend._callback_server is not None
        finally:
            backend.disconnect()
        backend._port = "not-a-port"
        assert backend._start_callback_server() is False
        assert backend._callback_server is None
        assert backend.connection_info.startswith("Weixin callback bind failed: ")

    def test_qq_start_and_bad_port(self) -> None:
        """QQ starts on an ephemeral port; a garbage port is caught."""
        from kiss.agents.third_party_agents.qq_agent import QQChannelBackend

        backend = QQChannelBackend()
        backend._port = "0"
        try:
            assert backend._start_webhook_server() is True
            assert backend._webhook_server is not None
        finally:
            backend.disconnect()
        backend._port = "not-a-port"
        assert backend._start_webhook_server() is False
        assert backend._webhook_server is None
        assert backend.connection_info.startswith("QQ webhook bind failed: ")

    def test_openai_compat_start_and_bind_conflict(self) -> None:
        """The API server reports its bound port in connection_info."""
        from kiss.agents.third_party_agents.openai_compat_agent import (
            OpenAICompatChannelBackend,
        )

        backend = OpenAICompatChannelBackend()
        backend._api_key = "test-key"
        backend._bind_host = "127.0.0.1"
        backend._port = 0
        try:
            assert backend._start_server() is True
            assert backend._server is not None
            bound = backend._server.server_address[1]
            assert backend.connection_info == (
                f"OpenAI-compatible API serving on 127.0.0.1:{bound}"
            )
        finally:
            backend.disconnect()
        blocker, port = _blocked_port()
        try:
            backend._port = port
            backend._bind_host = ""
            assert backend._start_server() is False
            assert backend._server is None
            assert backend.connection_info.startswith(
                "OpenAI-compatible API bind failed: "
            )
        finally:
            blocker.close()

    def test_a2a_start_and_bad_port(self) -> None:
        """The A2A server starts from string config and catches bad ports."""
        from kiss.agents.third_party_agents.a2a_agent import A2AChannelBackend

        backend = A2AChannelBackend()
        backend._bind_host = "127.0.0.1"
        backend._port = "0"
        try:
            assert backend._start_server() is True
            assert backend._server is not None
        finally:
            backend.disconnect()
        backend._port = "not-a-port"
        assert backend._start_server() is False
        assert backend._server is None
        assert backend.connection_info.startswith("A2A server bind failed: ")

    def test_webhook_start_and_bad_port(self) -> None:
        """The webhook server records its bound port and route count."""
        from kiss.agents.third_party_agents.webhook_agent import WebhookChannelBackend

        backend = WebhookChannelBackend()
        backend._port = "0"
        try:
            assert backend._start_server() is True
            assert backend._bound_port > 0
            assert backend.connection_info == (
                f"Webhook server listening on port {backend._bound_port} (0 route(s))"
            )
        finally:
            backend.disconnect()
        backend._port = "not-a-port"
        assert backend._start_server() is False
        assert backend._server is None
        assert backend.connection_info.startswith("Webhook server bind failed: ")
