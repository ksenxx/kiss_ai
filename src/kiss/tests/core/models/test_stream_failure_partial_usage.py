# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: usage observed before a streaming failure is billed once.

A provider bills a streamed generation as soon as it reports usage.  When
the stream then stalls (no terminal ``[DONE]`` / ``message_stop`` / EOF),
the adapter raises a retryable ``TimeoutError``; ``KISSAgent`` bills the
failed call through :meth:`Model.take_partial_usage_response`.  The
Chat Completions, Gemini and Anthropic adapters used to drop the usage
they had already seen, so the spend vanished from the task's cost.

Each test drives the real SDK against a real local SSE server (no mocks):
usage-bearing frames are written, then the connection is held open until
``stream_stall_timeout`` fires.  The observed usage must be returned by
``take_partial_usage_response()`` exactly once; after a successful turn
nothing may be left there (no double billing).
"""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from kiss.core import stop_signal
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.model import Model
from kiss.core.models.model_info import calculate_cost
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.print_to_console import ConsolePrinter
from kiss.tests.core.models.anthropic_sse_harness import sse, text_message_stream
from kiss.tests.core.models.gemini_sse_harness import (
    GeminiScript,
    chunk,
    serve,
    text_part,
)
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    chat_chunk,
)

_STALL_TIMEOUT = 1.0


class _Policy:
    """Answers with fixed SSE chunks, optionally holding the stream open."""

    def __init__(self, chunks: list[bytes], hold: bool) -> None:
        self.chunks = chunks
        self.release = threading.Event() if hold else None

    def __call__(self, request: Request) -> Reply:
        """Return the scripted reply."""
        return Reply(sse_chunks=self.chunks, hold=self.release)


@contextmanager
def _serve(chunks: list[bytes], hold: bool) -> Iterator[ScriptedOpenAIServer]:
    """Run a scripted SSE server, releasing any held stream on teardown."""
    policy = _Policy(chunks, hold)
    with ScriptedOpenAIServer(policy) as server:
        try:
            yield server
        finally:
            if policy.release is not None:
                policy.release.set()


def _assert_stall_billed_once(model: Model, expected: tuple[int, int]) -> None:
    """Generate, expect a stall, and check the observed usage is billed once."""
    with pytest.raises(TimeoutError):
        model.generate()
    partial = model.take_partial_usage_response()
    assert partial is not None, "usage seen before the stall was dropped"
    counts = model.extract_input_output_token_counts_from_response(partial)
    assert counts[:2] == expected
    assert model.take_partial_usage_response() is None


# ---------------------------------------------------------------- OpenAI

_OPENAI_MODEL = "gpt-partial-usage-under-test"


def _chat(choices: list[dict[str, Any]], usage: dict[str, int] | None = None) -> bytes:
    payload: dict[str, Any] = {
        "id": "chatcmpl-usage",
        "object": "chat.completion.chunk",
        "model": _OPENAI_MODEL,
        "choices": choices,
    }
    if usage is not None:
        payload["usage"] = usage
    return chat_chunk(payload)


_CHAT_CHUNKS = [
    _chat([{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": None}]),
    _chat([{"index": 0, "delta": {}, "finish_reason": "stop"}]),
    _chat([], usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}),
]


def _openai_model(server: ScriptedOpenAIServer) -> OpenAICompatibleModel:
    model = OpenAICompatibleModel(
        _OPENAI_MODEL,
        base_url=server.base_url,
        api_key="test-key",
        model_config={"stream_stall_timeout": _STALL_TIMEOUT},
        token_callback=lambda _t: None,
    )
    model.initialize("Say hi.")
    return model


def test_openai_stall_after_usage_bills_observed_usage() -> None:
    with _serve(_CHAT_CHUNKS, hold=True) as server:
        _assert_stall_billed_once(_openai_model(server), (11, 7))


def test_openai_success_leaves_no_partial_usage() -> None:
    with _serve([*_CHAT_CHUNKS, b"data: [DONE]\n\n"], hold=False) as server:
        model = _openai_model(server)
        text, response = model.generate()
        assert text == "hi"
        assert model.extract_input_output_token_counts_from_response(response)[:2] == (11, 7)
        assert model.take_partial_usage_response() is None


# ---------------------------------------------------------------- Gemini

_GEMINI_CHUNKS = [
    chunk(
        [text_part("hi")],
        usage={"promptTokenCount": 13, "candidatesTokenCount": 5, "totalTokenCount": 18},
    ),
]


@pytest.fixture
def gemini_endpoint() -> Generator[tuple[str, GeminiScript]]:
    """A real local Gemini endpoint for one test."""
    yield from serve()


def _gemini_model(monkeypatch: pytest.MonkeyPatch, base_url: str) -> GeminiModel:
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
    model = GeminiModel(
        "gemini-partial-usage-under-test",
        api_key="test-key",
        model_config={"stream_stall_timeout": _STALL_TIMEOUT},
        token_callback=lambda _t: None,
    )
    model.initialize("Say hi.")
    return model


def test_gemini_stall_after_usage_bills_observed_usage(
    monkeypatch: pytest.MonkeyPatch, gemini_endpoint: tuple[str, GeminiScript]
) -> None:
    base_url, script = gemini_endpoint
    script.play(_GEMINI_CHUNKS, after="silent")
    _assert_stall_billed_once(_gemini_model(monkeypatch, base_url), (13, 5))


def test_gemini_success_leaves_no_partial_usage(
    monkeypatch: pytest.MonkeyPatch, gemini_endpoint: tuple[str, GeminiScript]
) -> None:
    base_url, script = gemini_endpoint
    script.play(_GEMINI_CHUNKS, after="close")
    model = _gemini_model(monkeypatch, base_url)
    text, response = model.generate()
    assert text == "hi"
    assert model.extract_input_output_token_counts_from_response(response)[:2] == (13, 5)
    assert model.take_partial_usage_response() is None


# ---------------------------------------------------------------- Anthropic

_ANTHROPIC_MODEL = "claude-partial-usage-under-test"


def _anthropic_stream() -> list[bytes]:
    """A full turn with input 20 + cache read 6, and output 9 at message_delta."""
    chunks = text_message_stream("hi", _ANTHROPIC_MODEL)
    chunks[0] = chunks[0].replace(
        b'"usage": {"input_tokens": 3, "output_tokens": 1}',
        b'"usage": {"input_tokens": 20, "output_tokens": 1, "cache_read_input_tokens": 6}',
    )
    chunks[4] = sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 9},
        },
    )
    return chunks


def _anthropic_model(
    monkeypatch: pytest.MonkeyPatch, server: ScriptedOpenAIServer
) -> AnthropicModel:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", server.base_url.removesuffix("/v1"))
    model = AnthropicModel(
        _ANTHROPIC_MODEL,
        api_key="test-key",
        model_config={"stream_stall_timeout": _STALL_TIMEOUT},
        token_callback=lambda _t: None,
    )
    model.initialize("Say hi.")
    return model


def test_anthropic_stall_after_usage_bills_observed_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Everything up to message_delta, then silence instead of message_stop.
    with _serve(_anthropic_stream()[:5], hold=True) as server:
        model = _anthropic_model(monkeypatch, server)
        with pytest.raises(TimeoutError):
            model.generate()
        partial = model.take_partial_usage_response()
        assert partial is not None, "usage seen before the stall was dropped"
        counts = model.extract_input_output_token_counts_from_response(partial)
        assert counts[:3] == (20, 9, 6)
        assert model.take_partial_usage_response() is None


def test_anthropic_success_leaves_no_partial_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _serve(_anthropic_stream(), hold=False) as server:
        model = _anthropic_model(monkeypatch, server)
        text, response = model.generate()
        assert text == "hi"
        assert model.extract_input_output_token_counts_from_response(response)[:3] == (20, 9, 6)
        assert model.take_partial_usage_response() is None


# ---------------------------------------------------------------- Stop in KISSAgent

_PRICED_MODEL = "gpt-4o-mini"


def _stop_once_usage_seen(model: Model, stop: threading.Event) -> None:
    """Press Stop as soon as the adapter has observed the usage chunk."""
    deadline = time.monotonic() + 10.0
    while model._rejected_response is None and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()


def _run_agent_until_stop(model_name: str, is_agentic: bool) -> tuple[KISSAgent, Model]:
    """Run a real agent on a held stream and press Stop once usage arrives.

    Asserts the run ends with the Stop's ``KeyboardInterrupt``.
    """
    config = {"stream_stall_timeout": 10.0}
    with _serve(_CHAT_CHUNKS, hold=True) as server:
        model = OpenAICompatibleModel(
            model_name,
            base_url=server.base_url,
            api_key="test-key",
            model_config=dict(config),
        )
        agent = KISSAgent("stop-bills-partial-usage")
        agent.model = model
        stop = threading.Event()
        watcher = threading.Thread(target=_stop_once_usage_seen, args=(model, stop))
        stop_signal.set_thread_stop_event(stop)
        watcher.start()
        try:
            with pytest.raises(KeyboardInterrupt):
                agent.run(
                    model_name,
                    "Say hi.",
                    is_agentic=is_agentic,
                    model_config=dict(config),
                    # A printer gives the adapter a token callback, so it streams.
                    printer=ConsolePrinter(file=io.StringIO()),
                )
        finally:
            stop_signal.set_thread_stop_event(None)
            watcher.join()
    assert agent.model is model, "the pre-built adapter was not reused"
    return agent, model


@pytest.mark.parametrize("is_agentic", [False, True])
def test_agent_stop_after_usage_bills_observed_usage_once(is_agentic: bool) -> None:
    """A user Stop mid-stream bills the usage seen so far to the stopped run."""
    agent, model = _run_agent_until_stop(_PRICED_MODEL, is_agentic)
    assert agent.total_tokens_used == 18
    assert agent.budget_used == pytest.approx(calculate_cost(_PRICED_MODEL, 11, 7))
    assert model.take_partial_usage_response() is None


@pytest.mark.parametrize("is_agentic", [False, True])
def test_agent_stop_on_unpriced_model_still_raises_the_stop(is_agentic: bool) -> None:
    """Billing an unpriced model raises KISSError; the Stop must still win."""
    agent, model = _run_agent_until_stop(_OPENAI_MODEL, is_agentic)
    assert agent.budget_used == 0.0
    assert model.take_partial_usage_response() is None
