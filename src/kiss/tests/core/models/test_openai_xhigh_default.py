# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: gpt-5.5 family defaults to reasoning_effort='xhigh'.

A real ThreadingHTTPServer captures the JSON body of the Chat Completions
request so we can assert on what was actually sent over the wire — no
mocks, patches, or fakes.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kiss.core.models.model_info import model
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel


def _ok_chunk(content: str = "ok") -> str:
    """Build a single non-streaming chat.completion JSON body."""
    return json.dumps(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "gpt-5.5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            },
        }
    )


# Each handler instance is per-request; persist captured bodies on the
# class itself so the test can read them after .generate() returns.
class _CapturingHandler(BaseHTTPRequestHandler):
    """Captures the JSON body of every POST and returns a minimal response."""

    captured_bodies: list[dict[str, object]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            self.__class__.captured_bodies.append(json.loads(raw))
        except json.JSONDecodeError:
            self.__class__.captured_bodies.append({})

        body = _ok_chunk().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture
def capture_server() -> Generator[str]:
    _CapturingHandler.captured_bodies = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CapturingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()


class TestOpenAIXhighDefault:
    """gpt-5.5 family must send reasoning_effort='xhigh' by default."""

    def test_gpt_5_5_defaults_to_xhigh(self, capture_server: str) -> None:
        """A bare gpt-5.5 call must send reasoning_effort='xhigh'."""
        m = OpenAICompatibleModel(
            "gpt-5.5",
            base_url=capture_server,
            api_key="test-key",
        )
        m.initialize("hi")
        m.generate()
        assert _CapturingHandler.captured_bodies, "no request reached the server"
        body = _CapturingHandler.captured_bodies[-1]
        assert body.get("reasoning_effort") == "xhigh"

    def test_explicit_value_is_not_overridden(self, capture_server: str) -> None:
        """A caller-supplied reasoning_effort must win over the default."""
        m = OpenAICompatibleModel(
            "gpt-5.5",
            base_url=capture_server,
            api_key="test-key",
            model_config={"reasoning_effort": "high"},
        )
        m.initialize("hi")
        m.generate()
        body = _CapturingHandler.captured_bodies[-1]
        assert body.get("reasoning_effort") == "high"

    def test_caller_dict_is_not_mutated(self, capture_server: str) -> None:
        """Defaulting xhigh must NOT mutate the caller's model_config dict."""
        caller_config: dict[str, object] = {}
        OpenAICompatibleModel(
            "gpt-5.5",
            base_url=capture_server,
            api_key="test-key",
            model_config=caller_config,
        )
        assert caller_config == {}, (
            "OpenAICompatibleModel must not mutate the caller's model_config dict"
        )

    def test_gpt_5_5_dated_variant_defaults_to_xhigh(
        self, capture_server: str
    ) -> None:
        """Dated gpt-5.5 model names must also default to xhigh."""
        m = OpenAICompatibleModel(
            "gpt-5.5-2026-04-23",
            base_url=capture_server,
            api_key="test-key",
        )
        m.initialize("hi")
        m.generate()
        body = _CapturingHandler.captured_bodies[-1]
        assert body.get("reasoning_effort") == "xhigh"

    def test_gpt_5_5_pro_does_not_default(self, capture_server: str) -> None:
        """gpt-5.5-pro does not accept reasoning_effort and must NOT default."""
        m = OpenAICompatibleModel(
            "gpt-5.5-pro",
            base_url=capture_server,
            api_key="test-key",
        )
        m.initialize("hi")
        m.generate()
        body = _CapturingHandler.captured_bodies[-1]
        assert "reasoning_effort" not in body

    def test_gpt_5_does_not_default(self, capture_server: str) -> None:
        """gpt-5 only supports up to 'high' — must NOT be defaulted to xhigh."""
        m = OpenAICompatibleModel(
            "gpt-5",
            base_url=capture_server,
            api_key="test-key",
        )
        m.initialize("hi")
        m.generate()
        body = _CapturingHandler.captured_bodies[-1]
        assert "reasoning_effort" not in body

    def test_gpt_4o_does_not_default(self, capture_server: str) -> None:
        """Non-reasoning OpenAI models must NOT have reasoning_effort added."""
        m = OpenAICompatibleModel(
            "gpt-4o",
            base_url=capture_server,
            api_key="test-key",
        )
        m.initialize("hi")
        m.generate()
        body = _CapturingHandler.captured_bodies[-1]
        assert "reasoning_effort" not in body

    def test_openrouter_gpt_5_5_defaults_to_xhigh(
        self, capture_server: str
    ) -> None:
        """openrouter/openai/gpt-5.5 (after openrouter/ strip) defaults too."""
        m = OpenAICompatibleModel(
            "openrouter/openai/gpt-5.5",
            base_url=capture_server,
            api_key="test-key",
        )
        m.initialize("hi")
        m.generate()
        body = _CapturingHandler.captured_bodies[-1]
        assert body.get("reasoning_effort") == "xhigh"

    def test_model_factory_routes_gpt_5_5_with_xhigh_default(
        self, capture_server: str
    ) -> None:
        """The model() factory path must also default to xhigh."""
        m = model(
            "gpt-5.5",
            model_config={"base_url": capture_server, "api_key": "test-key"},
        )
        m.initialize("hi")
        m.generate()
        body = _CapturingHandler.captured_bodies[-1]
        assert body.get("reasoning_effort") == "xhigh"

    def test_openrouter_gpt_latest_alias_defaults_to_xhigh(
        self, capture_server: str
    ) -> None:
        """openrouter/~openai/gpt-latest (alias for current best GPT) defaults too."""
        m = OpenAICompatibleModel(
            "openrouter/~openai/gpt-latest",
            base_url=capture_server,
            api_key="test-key",
        )
        m.initialize("hi")
        m.generate()
        body = _CapturingHandler.captured_bodies[-1]
        assert body.get("reasoning_effort") == "xhigh"

    def test_unknown_model_does_not_default(self, capture_server: str) -> None:
        """A model name not present in MODEL_INFO must NOT receive xhigh.

        This guards custom local endpoints (Ollama, vLLM, LM Studio) and
        any future model whose entry has not yet been added — defaulting
        ``reasoning_effort`` for an unknown model would be wrong because
        the upstream API may reject the parameter.
        """
        m = OpenAICompatibleModel(
            "some-custom-local-model-not-in-model-info",
            base_url=capture_server,
            api_key="test-key",
        )
        m.initialize("hi")
        m.generate()
        body = _CapturingHandler.captured_bodies[-1]
        assert "reasoning_effort" not in body

    def test_model_info_flag_drives_defaulting(self, capture_server: str) -> None:
        """The xhigh default must be driven solely by the MODEL_INFO flag.

        Flipping ``supports_xhigh_reasoning_effort`` on a model that does NOT
        normally default (here: ``gpt-4o``) must immediately cause the next
        ``OpenAICompatibleModel`` instance built for that model to send
        ``reasoning_effort: xhigh``.  This is the contract that makes adding
        future xhigh-capable models a one-line change in ``model_info.py``.
        """
        from kiss.core.models.model_info import MODEL_INFO
        original = MODEL_INFO["gpt-4o"].supports_xhigh_reasoning_effort
        MODEL_INFO["gpt-4o"].supports_xhigh_reasoning_effort = True
        try:
            m = OpenAICompatibleModel(
                "gpt-4o",
                base_url=capture_server,
                api_key="test-key",
            )
            m.initialize("hi")
            m.generate()
            body = _CapturingHandler.captured_bodies[-1]
            assert body.get("reasoning_effort") == "xhigh"
        finally:
            MODEL_INFO["gpt-4o"].supports_xhigh_reasoning_effort = original
