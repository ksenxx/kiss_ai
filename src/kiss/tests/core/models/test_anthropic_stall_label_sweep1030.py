# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""An Anthropic stall still names the model, through the shared stop-or-stall classifier.

``AnthropicModel`` classified a timeout with its own copy of
``stream_abort.stop_or_stall_error`` and a ``_stall_error`` that put the
model's name in front of the shared stall message.  It now calls
``stop_or_stall_error`` with that name as the *label*.  A real local
Messages endpoint sends the response headers and then nothing, so the
stream's watchdog aborts it as stalled.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.stream_abort import is_stall_error

_MODEL = "claude-stall-label-under-test"


class _HeadersThenSilence(BaseHTTPRequestHandler):
    """Sends a streaming response's headers, then nothing until released."""

    release = threading.Event()

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.flush()
        self.release.wait(30)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class _DaemonServer(ThreadingHTTPServer):
    daemon_threads = True


def test_a_stalled_anthropic_stream_names_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The retryable stall error starts with the model's name and is still a stall."""
    release = threading.Event()
    handler = type("_Handler", (_HeadersThenSilence,), {"release": release})
    server = _DaemonServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    model = AnthropicModel(_MODEL, api_key="local", model_config={"stream_stall_timeout": 1.0})
    model.initialize("Say hello.")
    try:
        with pytest.raises(TimeoutError) as caught:
            model.generate()
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(5)

    assert str(caught.value).startswith(
        f"Anthropic model {_MODEL}: Model stream stalled: no data received for 1s"
    ), str(caught.value)
    assert is_stall_error(caught.value)
