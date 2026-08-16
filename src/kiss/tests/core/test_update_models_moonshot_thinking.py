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

MOONSHOT_LEVELS = ("low", "high", "max")


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


ALL_KNOWN_LEVELS = ("low", "medium", "high", "xhigh", "max")


class TestRuntimeAliasResolution:
    """Kimi K3 aliases must resolve and route like every other alias."""

    def test_strip_thinking_alias_maps_max_to_base(self) -> None:
        from kiss.core.models.model_info import MODEL_INFO, _strip_thinking_alias

        for level in MOONSHOT_LEVELS:
            name = f"kimi-k3-{level}"
            assert name in MODEL_INFO, f"{name} missing from loaded MODEL_INFO"
            assert _strip_thinking_alias(name) == "kimi-k3"

    def test_provider_model_name_and_effort_for_max_alias(self) -> None:
        from kiss.core.models.openai_compatible_model import (
            _model_thinking_level,
            _provider_model_name,
        )

        assert _provider_model_name("kimi-k3-max") == "kimi-k3"
        assert _model_thinking_level("kimi-k3-max") == "max"
        assert (
            _provider_model_name("openrouter/moonshotai/kimi-k3-max")
            == "moonshotai/kimi-k3"
        )
        assert _model_thinking_level("openrouter/moonshotai/kimi-k3-max") == "max"

    def test_unmarked_max_suffix_is_not_stripped(self) -> None:
        """A name ending in -max with no catalog alias marker stays intact."""
        from kiss.core.models.model_info import _strip_thinking_alias

        assert _strip_thinking_alias("acme/custom-max") == "acme/custom-max"

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

    def test_kimi_aliases_cost_the_same_as_base(self) -> None:
        from kiss.core.models.model_info import calculate_cost

        base_cost = calculate_cost("kimi-k3", 1000, 500, 200, 100)
        for level in MOONSHOT_LEVELS:
            assert calculate_cost(f"kimi-k3-{level}", 1000, 500, 200, 100) == base_cost


def test_every_bundled_alias_thinking_is_a_known_level() -> None:
    """Every marked alias in the bundled catalog uses a known level name."""
    from kiss.core.models.model_info import MODEL_INFO

    for name, info in MODEL_INFO.items():
        if info.alias_of:
            assert info.thinking in ALL_KNOWN_LEVELS, (name, info.thinking)
