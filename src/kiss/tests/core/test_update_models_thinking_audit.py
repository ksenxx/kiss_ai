# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Cross-vendor regression audit for reasoning-effort alias generation.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.scripts.test_update_models_thinking_audit``; the non-core tests remain there.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class _EffortCaptureHandler(BaseHTTPRequestHandler):
    """Generic OpenAI-compatible chat-completions emulator.

    Captures every request body into ``captured_bodies`` and answers with
    a minimal successful completion, echoing the requested model id.
    Serves the OpenRouter and Together routes alike (both speak the same
    ``/chat/completions`` wire dialect).
    """

    captured_bodies: list[dict] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.captured_bodies.append(body)
        payload = json.dumps(
            {
                "id": "cmpl-wire",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", ""),
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
def effort_wire(monkeypatch: pytest.MonkeyPatch) -> Generator[str]:
    """Route the openrouter and together vendors at an in-process emulator."""
    import dataclasses

    from kiss.core import config as config_module
    from kiss.core.models import model_info

    _EffortCaptureHandler.captured_bodies = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EffortCaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    providers = tuple(
        dataclasses.replace(p, base_url=base_url)
        if p.name in ("openrouter", "together")
        else p
        for p in model_info.OPENAI_COMPATIBLE_PROVIDERS
    )
    monkeypatch.setattr(model_info, "OPENAI_COMPATIBLE_PROVIDERS", providers)
    monkeypatch.setattr(
        config_module.DEFAULT_CONFIG, "OPENROUTER_API_KEY", "test-key", raising=False
    )
    monkeypatch.setattr(
        config_module.DEFAULT_CONFIG, "TOGETHER_API_KEY", "test-key", raising=False
    )
    try:
        yield base_url
    finally:
        server.shutdown()


def _generate_and_capture(alias: str) -> dict:
    """Run a real ``generate()`` through ``alias`` and return the wire body."""
    from kiss.core.models.model_info import model as create_model

    m = create_model(alias)
    m.initialize("Say hello in one word.")
    text, _ = m.generate()
    assert text.strip() == "hello"
    assert len(_EffortCaptureHandler.captured_bodies) == 1, (
        f"Exactly one request expected for {alias}; "
        f"got {len(_EffortCaptureHandler.captured_bodies)}"
    )
    return _EffortCaptureHandler.captured_bodies[0]


class TestWireShapePerVendorRoute:
    """Each new family's alias must put ``reasoning_effort`` on the wire."""

    def test_grok_4_5_high_via_openrouter(self, effort_wire: str) -> None:
        """openrouter/x-ai/grok-4.5-high → model=x-ai/grok-4.5, effort=high."""
        body = _generate_and_capture("openrouter/x-ai/grok-4.5-high")
        assert body["model"] == "x-ai/grok-4.5", "Wire id must be the base model"
        assert body["reasoning_effort"] == "high"

    def test_glm_5_2_max_via_together(self, effort_wire: str) -> None:
        """zai-org/GLM-5.2-max → model=zai-org/GLM-5.2, effort=max."""
        body = _generate_and_capture("zai-org/GLM-5.2-max")
        assert body["model"] == "zai-org/GLM-5.2", "Wire id must be the base model"
        assert body["reasoning_effort"] == "max"

    def test_glm_5_2_max_via_openrouter(self, effort_wire: str) -> None:
        """openrouter/z-ai/glm-5.2-max → model=z-ai/glm-5.2, effort=max."""
        body = _generate_and_capture("openrouter/z-ai/glm-5.2-max")
        assert body["model"] == "z-ai/glm-5.2", "Wire id must be the base model"
        assert body["reasoning_effort"] == "max"

    def test_gpt_oss_120b_medium_via_together(self, effort_wire: str) -> None:
        """openai/gpt-oss-120b-medium → model=openai/gpt-oss-120b, effort=medium."""
        body = _generate_and_capture("openai/gpt-oss-120b-medium")
        assert body["model"] == "openai/gpt-oss-120b", "Wire id must be the base model"
        assert body["reasoning_effort"] == "medium"

    def test_gpt_oss_20b_low_via_openrouter(self, effort_wire: str) -> None:
        """openrouter/openai/gpt-oss-20b-low → model=openai/gpt-oss-20b, effort=low."""
        body = _generate_and_capture("openrouter/openai/gpt-oss-20b-low")
        assert body["model"] == "openai/gpt-oss-20b", "Wire id must be the base model"
        assert body["reasoning_effort"] == "low"
