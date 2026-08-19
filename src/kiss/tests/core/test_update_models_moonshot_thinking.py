# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for Moonshot/Kimi reasoning-effort alias generation.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.scripts.test_update_models_moonshot_thinking``; the non-core tests remain there.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class _MoonshotHandler(BaseHTTPRequestHandler):
    """Emulates api.moonshot.ai/v1/chat/completions for kimi-k3.

    Accepts ``reasoning_effort`` values in ``accepted_efforts`` (and
    requests without the field), rejects every other value with HTTP 400
    — exactly Moonshot's documented K3 behavior. Tests may shrink
    ``accepted_efforts`` to exercise the probe's descending fallback.
    """

    captured_efforts: list[object] = []
    captured_bodies: list[dict] = []
    accepted_efforts: tuple[str, ...] = ("low", "high", "max")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        effort = body.get("reasoning_effort")
        self.__class__.captured_efforts.append(effort)
        self.__class__.captured_bodies.append(body)
        if effort is not None and effort not in self.__class__.accepted_efforts:
            payload = json.dumps(
                {
                    "error": {
                        "message": f"Invalid value for reasoning_effort: {effort}",
                        "type": "invalid_request_error",
                    }
                }
            ).encode()
            self.send_response(400)
        elif body.get("stream"):
            chunk = {
                "id": "cmpl-k3",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "kimi-k3",
                "choices": [
                    {"index": 0, "delta": {"content": "hello"}, "finish_reason": None}
                ],
            }
            done = {
                "id": "cmpl-k3",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "kimi-k3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
            payload = (
                f"data: {json.dumps(chunk)}\n\n"
                f"data: {json.dumps(done)}\n\n"
                "data: [DONE]\n\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        else:
            payload = json.dumps(
                {
                    "id": "cmpl-k3",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "kimi-k3",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "hello"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "total_tokens": 4,
                    },
                }
            ).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


@pytest.fixture
def moonshot_wire(monkeypatch: pytest.MonkeyPatch) -> Generator[str]:
    """Route the registered moonshot vendor at an in-process K3 emulator."""
    import dataclasses

    from kiss.core import config as config_module
    from kiss.core.models import model_info

    _MoonshotHandler.captured_efforts = []
    _MoonshotHandler.captured_bodies = []
    _MoonshotHandler.accepted_efforts = ("low", "high", "max")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MoonshotHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    providers = tuple(
        dataclasses.replace(p, base_url=base_url) if p.name == "moonshot" else p
        for p in model_info.OPENAI_COMPATIBLE_PROVIDERS
    )
    monkeypatch.setattr(model_info, "OPENAI_COMPATIBLE_PROVIDERS", providers)
    monkeypatch.setattr(
        config_module.DEFAULT_CONFIG, "MOONSHOT_API_KEY", "test-key", raising=False
    )
    try:
        yield base_url
    finally:
        server.shutdown()


class TestRuntimeAliasResolution:
    """Kimi K3 aliases must resolve and route like every other alias."""

    def test_factory_builds_kimi_max_alias_with_effort(self, moonshot_wire: str) -> None:
        """model('kimi-k3-max') must send model=kimi-k3, effort=max."""
        from kiss.core.models.model_info import model as create_model

        m = create_model("kimi-k3-max")
        assert m.model_config.get("reasoning_effort") == "max"
        assert getattr(m, "_api_model_name", None) == "kimi-k3"

    def test_alias_generate_sends_base_model_and_max_on_the_wire(
        self, moonshot_wire: str
    ) -> None:
        """A real generate() through the alias must carry both wire fields."""
        from kiss.core.models.model_info import model as create_model

        m = create_model("kimi-k3-max")
        m.initialize("Say hello in one word.")
        text, _ = m.generate()

        assert text.strip() == "hello"
        assert len(_MoonshotHandler.captured_bodies) == 1
        body = _MoonshotHandler.captured_bodies[0]
        assert body["model"] == "kimi-k3", "The wire id must be the base model"
        assert body["reasoning_effort"] == "max"
