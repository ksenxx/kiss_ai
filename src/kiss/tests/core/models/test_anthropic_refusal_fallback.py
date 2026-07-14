# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: Anthropic safety refusals must trigger model fallback.

Bug reproduction (production failures at 2026-07-14 11:35-11:47 in
``~/.kiss/sorcar.db``, tasks ``daa89a7e...`` and ``c3cd9c95...``, model =
``claude-fable-5``):

* On a benign security-research prompt (the SWEdefend paper task),
  fable-5's safety layer returned ``stop_reason="refusal"`` with an EMPTY
  ``content`` list (usage ``in=1 out=9``) — verified by a full-fidelity
  live-API replay of the exact production request.  ``claude-opus-4-8``
  answered the identical request normally (thinking + tool_use).
* ``AnthropicModel.generate_and_process_with_tools`` ignored the
  ``refusal`` stop reason and returned ``([], "", response)``.
* ``KISSAgent._execute_step`` counted the empty turn, burned a useless
  ``"MUST have at least one function call"`` retry (refusals are
  deterministic for identical content, so the retry was refused too),
  then raised ``_EmptyModelResponseError`` blaming "a streaming or
  reasoning-block parsing issue in the model adapter" — a misdiagnosis —
  and fell back with the misleading reason "repeated empty responses".

Fix under test:

1. ``AnthropicModel._raise_on_refusal`` raises ``ModelRefusalError`` when
   ``response.stop_reason == "refusal"`` (both ``generate`` and
   ``generate_and_process_with_tools``).
2. ``KISSAgent._run_agentic_loop`` catches ``ModelRefusalError`` and
   immediately swaps to the registered fallback model (no wasted retry
   turn), announcing ``a safety refusal (stop_reason="refusal")``.

Test strategy (no mocks, patches, or fakes): a local
``ThreadingHTTPServer`` speaks the real Anthropic SSE wire format to a
real ``anthropic`` SDK client; it replays the exact refusal stream shape
the live API produced for the failing production request (empty content,
``stop_reason="refusal"``) for the primary model and a normal ``finish``
tool-use stream for the fallback model.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import anthropic
import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError, ModelRefusalError
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model_info import MODEL_INFO, ModelInfo

_PRIMARY = "claude-refusal-primary-under-test"
_FALLBACK = "claude-refusal-fallback-under-test"

_OPENAI_FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Finish the task",
        "parameters": {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        },
    },
}


def _sse(event_type: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


def _refusal_stream(model_name: str) -> list[bytes]:
    """The exact stream shape fable-5 produced for the failing request:
    an assistant message with ZERO content blocks and
    ``stop_reason="refusal"`` (live-API usage was ``in=1 out=9``)."""
    return [
        _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_refusal",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model_name,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            },
        ),
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "refusal", "stop_sequence": None},
                "usage": {"output_tokens": 9},
            },
        ),
        _sse("message_stop", {"type": "message_stop"}),
    ]


def _finish_tool_stream(model_name: str) -> list[bytes]:
    """A normal agentic turn: one ``finish`` tool_use block."""
    return [
        _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_finish",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model_name,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            },
        ),
        _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_finish",
                    "name": "finish",
                    "input": {},
                },
            },
        ),
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"result": "done"}',
                },
            },
        ),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 20},
            },
        ),
        _sse("message_stop", {"type": "message_stop"}),
    ]


_SEEN_MODELS: list[str] = []


class _RefusalHandler(BaseHTTPRequestHandler):
    """Refuses the primary model's requests; lets the fallback finish."""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        model_name = body.get("model", "")
        _SEEN_MODELS.append(model_name)
        if model_name == _FALLBACK:
            chunks = _finish_tool_stream(model_name)
        else:
            chunks = _refusal_stream(model_name)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(chunk)
            self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture(scope="module")
def refusal_server() -> Generator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RefusalHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


class TestAdapterRaisesOnRefusal:
    """``AnthropicModel`` must raise ``ModelRefusalError`` on a refusal."""

    def _make_model(self, refusal_server: str, name: str = "claude-fable-5") -> AnthropicModel:
        m = AnthropicModel(name, api_key="test-key")
        m.client = anthropic.Anthropic(api_key="test-key", base_url=refusal_server)
        m.conversation = [{"role": "user", "content": "Update the SWEdefend paper."}]
        return m

    def test_tools_turn_raises_model_refusal_error(self, refusal_server: str) -> None:
        """The agentic (tools) path must raise instead of returning an
        empty ``([], "", response)`` turn."""
        _SEEN_MODELS.clear()
        m = self._make_model(refusal_server)
        with pytest.raises(ModelRefusalError) as excinfo:
            m.generate_and_process_with_tools({}, tools_schema=[_OPENAI_FINISH_TOOL])
        msg = str(excinfo.value)
        assert 'stop_reason="refusal"' in msg
        assert "claude-fable-5" in msg

    def test_plain_generate_raises_model_refusal_error(self, refusal_server: str) -> None:
        """The no-tools path must raise too."""
        m = self._make_model(refusal_server)
        with pytest.raises(ModelRefusalError):
            m.generate()

    def test_refused_turn_is_not_appended_to_conversation(
        self, refusal_server: str
    ) -> None:
        """A refusal must not leave a (possibly empty) assistant turn in the
        conversation — the fallback model replays the same history."""
        m = self._make_model(refusal_server)
        before = [dict(msg) for msg in m.conversation]
        with pytest.raises(ModelRefusalError):
            m.generate_and_process_with_tools({}, tools_schema=[_OPENAI_FINISH_TOOL])
        assert m.conversation == before

    def test_refusal_error_is_a_kiss_error(self) -> None:
        """``_run_agentic_loop`` catches ``KISSError``; the refusal must be
        one so the fallback branch (not the generic retry) handles it."""
        assert issubclass(ModelRefusalError, KISSError)


def _ensure_api_key(monkeypatch: Any) -> None:
    """Give the ``model()`` factory a non-empty Anthropic key.

    ``AnthropicModel.initialize`` constructs a real ``Anthropic`` client
    with ``DEFAULT_CONFIG.ANTHROPIC_API_KEY``; on CI machines without the
    env var the empty string would make the SDK constructor fail before
    the test's ``pre_step_hook`` can point the client at the local server.
    """
    from kiss.core import config as config_module

    if not getattr(config_module.DEFAULT_CONFIG, "ANTHROPIC_API_KEY", ""):
        monkeypatch.setattr(
            config_module.DEFAULT_CONFIG,
            "ANTHROPIC_API_KEY",
            "test-key",
            raising=False,
        )


def _register_refusal_pair(monkeypatch: Any) -> None:
    """Register a synthetic ``claude-*`` primary/fallback pair in MODEL_INFO.

    The ``claude-`` prefix routes both through the real ``AnthropicModel``
    (see ``model_info.model``); ``extended_thinking=False`` keeps the wire
    request minimal.  The real ``claude-fable-5`` entry is left untouched so
    the test never depends on Anthropic credentials.
    """
    for name, fb in ((_PRIMARY, _FALLBACK), (_FALLBACK, None)):
        monkeypatch.setitem(
            MODEL_INFO,
            name,
            ModelInfo(
                context_length=128_000,
                input_price_per_million=0.0,
                output_price_per_million=0.0,
                is_function_calling_supported=True,
                is_embedding_supported=False,
                is_generation_supported=True,
                fallback=fb,
                extended_thinking=False,
            ),
        )


class TestAgentFallsBackOnRefusal:
    """KISSAgent must swap to the fallback model on a safety refusal."""

    def _point_clients_at(self, server_url: str) -> Any:
        def hook(m: Any) -> None:
            base = str(getattr(getattr(m, "client", None), "base_url", ""))
            if isinstance(m, AnthropicModel) and server_url not in base:
                m.client = anthropic.Anthropic(api_key="test-key", base_url=server_url)

        return hook

    def test_refusal_switches_to_fallback_and_finishes(
        self, monkeypatch: Any, refusal_server: str
    ) -> None:
        """Primary refuses on turn 1 → immediate swap (no wasted
        "MUST have at least one function call" retry) → fallback finishes."""
        _register_refusal_pair(monkeypatch)
        _ensure_api_key(monkeypatch)
        _SEEN_MODELS.clear()
        agent = KISSAgent("test-refusal-fallback")
        agent.pre_step_hook = self._point_clients_at(refusal_server)
        result = agent.run(
            model_name=_PRIMARY,
            prompt_template="Update the SWEdefend paper.",
            max_steps=5,
            max_budget=1.0,
            verbose=False,
        )
        assert result == "done"
        assert agent.model_name == _FALLBACK
        assert agent._fallback_used is True
        # Exactly ONE refused request to the primary (no useless retry
        # turn — the old empty-turn path sent two), then the fallback.
        assert _SEEN_MODELS == [_PRIMARY, _FALLBACK]

    def test_refusal_without_fallback_raises_refusal_error(
        self, monkeypatch: Any, refusal_server: str
    ) -> None:
        """With no fallback registered the refusal must surface as the
        actionable ``ModelRefusalError`` (not a misleading "repeated empty
        responses" diagnostic)."""
        primary = "claude-refusal-no-fallback-under-test"
        monkeypatch.setitem(
            MODEL_INFO,
            primary,
            ModelInfo(
                context_length=128_000,
                input_price_per_million=0.0,
                output_price_per_million=0.0,
                is_function_calling_supported=True,
                is_embedding_supported=False,
                is_generation_supported=True,
                fallback=None,
                extended_thinking=False,
            ),
        )
        _ensure_api_key(monkeypatch)
        _SEEN_MODELS.clear()
        agent = KISSAgent("test-refusal-no-fallback")
        agent.pre_step_hook = self._point_clients_at(refusal_server)
        with pytest.raises(ModelRefusalError) as excinfo:
            agent.run(
                model_name=primary,
                prompt_template="Update the SWEdefend paper.",
                max_steps=5,
                max_budget=1.0,
                verbose=False,
            )
        assert 'stop_reason="refusal"' in str(excinfo.value)
