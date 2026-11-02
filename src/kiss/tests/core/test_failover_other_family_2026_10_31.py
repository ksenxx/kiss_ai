# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: a filtered, refused or cut-off request fails over to another model family.

Proposal 6 of ``reports/kiss-sorcar-general-improvements.html``.  A
content-filter stop, a safety refusal and an output-cap cut-off are
deterministic for the same request and shared by every model of the
family (the same model's OpenRouter twin included), so retrying them on
the same family only burns steps.  Under test:

* the adapters type the stop: Responses ``incomplete_details.reason``
  ``content_filter`` -> :class:`ContentFilterError`, ``max_output_tokens``
  -> :class:`OutputCapError`; Chat Completions ``finish_reason``
  ``content_filter`` / ``length`` likewise; Anthropic ``stop_reason ==
  "max_tokens"`` with no complete ``tool_use`` block -> ``OutputCapError``
  (its partial usage stays billable, the conversation untouched);
* the agentic loop retries a cut-off request once at reasoning effort
  ``low`` (the request body carries it), then switches to a model of
  another family, with a visible notice naming both models and the stop
  reason; a filtered request switches at once; when only the same
  family is configured the notice says so;
* the billed failed call's ``llm_call`` event carries the ``stop_reason``.

Scripted local servers speak the OpenAI (Responses and Chat Completions)
and Anthropic wire formats to the real SDK clients; no mocks.  The
synthetic catalog entries declare a fallback of another family, so the
choice does not depend on which providers this machine has keys for.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import anthropic
import pytest

from kiss.core import config as config_module
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import ContentFilterError, KISSError, OutputCapError
from kiss.core.models import model_families
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model_info import MODEL_INFO, ModelInfo
from kiss.core.models.openai_compatible_model import OpenAICompatibleBase, OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.core.printer import Printer
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    chat_chunk,
    responses_event,
)

_OPENAI_PRIMARY = "gpt-failover-primary-under-test"
_OPENAI_TWIN = "gpt-failover-twin-under-test"
_ANTHROPIC_OTHER = "claude-failover-other-under-test"
_ANTHROPIC_PRIMARY = "claude-failover-primary-under-test"
_OPENAI_OTHER = "gpt-failover-other-under-test"

_FINISH_TOOL = {
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


class CapturePrinter(Printer):
    """Records every event the agent prints.

    Args:
        streaming: Whether to keep a token callback, which makes the
            adapters stream; without one they send blocking requests.
    """

    def __init__(self, streaming: bool = False) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        if not streaming:
            self.token_callback = None  # type: ignore[method-assign,assignment]

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> None:  # noqa: A002
        self.events.append({"type": type, "content": content, **kwargs})

    def token_callback(self, token: str) -> None:  # type: ignore[no-redef]
        pass

    def reset(self) -> None:
        pass

    def notices(self) -> list[str]:
        """The ``system_prompt`` texts: the fail-over and retry notices."""
        return [str(e["content"]) for e in self.events if e["type"] == "system_prompt"]

    def llm_calls(self) -> list[dict[str, Any]]:
        """The ``llm_call`` events, one per billed model call."""
        return [e for e in self.events if e["type"] == "llm_call"]


def _register(
    monkeypatch: pytest.MonkeyPatch, name: str, fallback: str | None, **extra: Any
) -> None:
    """Register a synthetic, free catalog entry for the duration of one test."""
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
            fallback=fallback,
            **extra,
        ),
    )


def _ensure_key(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Give the ``model()`` factory a non-empty provider key on machines without one."""
    if not getattr(config_module.DEFAULT_CONFIG, name, ""):
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, name, "test-key", raising=False)


# --- OpenAI wire replies -------------------------------------------------------


def _responses_usage() -> dict[str, Any]:
    return {
        "input_tokens": 700,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 300,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 1000,
    }


def _responses_incomplete(reason: str, model: str) -> Reply:
    """A billed Responses-API answer the provider ended early for *reason*."""
    return Reply(
        json_body={
            "id": "resp_incomplete",
            "object": "response",
            "created_at": 0,
            "model": model,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "incomplete",
            "incomplete_details": {"reason": reason},
            "output": [],
            "usage": _responses_usage(),
        }
    )


def _responses_finish(model: str) -> Reply:
    """A completed Responses-API turn calling ``finish``."""
    return Reply(
        json_body={
            "id": "resp_finish",
            "object": "response",
            "created_at": 0,
            "model": model,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_finish",
                    "name": "finish",
                    "arguments": json.dumps({"result": "done"}),
                    "status": "completed",
                }
            ],
            "usage": _responses_usage(),
        }
    )


def _chat_usage() -> dict[str, Any]:
    return {"prompt_tokens": 700, "completion_tokens": 300, "total_tokens": 1000}


def _chat_cut_short(finish_reason: str, model: str) -> Reply:
    """A blocking Chat Completions answer cut short with *finish_reason*."""
    return Reply(
        json_body={
            "id": "chatcmpl-cut",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "partial"},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": _chat_usage(),
        }
    )


def _chat_finish(model: str) -> Reply:
    """A Chat Completions turn calling ``finish``."""
    return Reply(
        json_body={
            "id": "chatcmpl-finish",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_finish",
                                "type": "function",
                                "function": {
                                    "name": "finish",
                                    "arguments": json.dumps({"result": "done"}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": _chat_usage(),
        }
    )


def _chat_streamed_length(model: str) -> Reply:
    """A streamed Chat Completions answer: text, ``finish_reason="length"``, then usage.

    The usage chunk carries no choices, as OpenAI sends it, so the billed
    partial response names no finish reason by itself.
    """
    base = {"id": "chatcmpl-stream", "object": "chat.completion.chunk", "created": 0}
    base["model"] = model
    text = {"index": 0, "delta": {"content": "part"}, "finish_reason": None}
    cut = {"index": 0, "delta": {}, "finish_reason": "length"}
    return Reply(
        sse_chunks=[
            chat_chunk({**base, "choices": [text]}),
            chat_chunk({**base, "choices": [cut]}),
            chat_chunk({**base, "choices": [], "usage": _chat_usage()}),
            b"data: [DONE]\n\n",
        ]
    )


def _run_openai(
    server: ScriptedOpenAIServer,
    printer: CapturePrinter,
    model_name: str,
    *,
    responses_api: bool,
    max_steps: int = 6,
) -> str:
    """Run an agent on *model_name* against the scripted server.

    Both the primary and a declared fallback keep the endpoint override,
    so a declared fallback of another family lands on the same server
    under its own model name.
    """
    agent = KISSAgent("failover-under-test")
    return agent.run(
        model_name=model_name,
        prompt_template="Do the task.",
        max_steps=max_steps,
        max_budget=1.0,
        verbose=False,
        printer=printer,
        model_config={
            "base_url": server.base_url,
            "api_key": "sk-test",
            "use_responses_api": responses_api,
        },
    )


def _models_seen(server: ScriptedOpenAIServer) -> list[str]:
    return [str(r.body.get("model")) for r in server.requests]


@pytest.fixture
def no_other_family(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the machine look as if no model of another family were configured."""
    monkeypatch.setattr(model_families, "SECOND_OPINION_MODELS", ())


# --- Responses API: content filter and output cap ------------------------------


def test_content_filter_switches_to_another_family_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``content_filter`` stop is never retried on the same model: the second
    request already goes to the declared other-family fallback, the notice names
    both models and the stop reason, and the filtered call's bill carries it."""
    _register(monkeypatch, _OPENAI_PRIMARY, _ANTHROPIC_OTHER, thinking="high")
    _register(monkeypatch, _ANTHROPIC_OTHER, None)

    def respond(request: Request) -> Reply:
        model = str(request.body.get("model"))
        if model == _OPENAI_PRIMARY:
            return _responses_incomplete("content_filter", model)
        return _responses_finish(model)

    printer = CapturePrinter()
    with ScriptedOpenAIServer(respond) as server:
        result = _run_openai(server, printer, _OPENAI_PRIMARY, responses_api=True)
        seen = _models_seen(server)

    assert result == "done"
    assert seen == [_OPENAI_PRIMARY, _ANTHROPIC_OTHER]
    assert printer.notices() == [
        f"Switched model from {_OPENAI_PRIMARY} to {_ANTHROPIC_OTHER} after a content "
        'filter stop (stop_reason="content_filter")'
    ]
    calls = printer.llm_calls()
    assert [c["model"] for c in calls] == [_OPENAI_PRIMARY, _ANTHROPIC_OTHER]
    assert calls[0]["stop_reason"] == "content_filter"
    assert calls[0]["input_tokens"] == 700 and calls[0]["output_tokens"] == 300


def test_output_cap_is_retried_once_at_low_effort_then_switches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``max_output_tokens`` gets one retry of the same request at effort ``low``
    (the request body says so; the first request ran at the model's own level);
    when that is cut short too, the run moves to the other family."""
    _register(monkeypatch, _OPENAI_PRIMARY, _ANTHROPIC_OTHER, thinking="high")
    _register(monkeypatch, _ANTHROPIC_OTHER, None)

    def respond(request: Request) -> Reply:
        model = str(request.body.get("model"))
        if model == _OPENAI_PRIMARY:
            return _responses_incomplete("max_output_tokens", model)
        return _responses_finish(model)

    printer = CapturePrinter()
    with ScriptedOpenAIServer(respond) as server:
        result = _run_openai(server, printer, _OPENAI_PRIMARY, responses_api=True)
        seen = _models_seen(server)
        efforts = [(r.body.get("reasoning") or {}).get("effort") for r in server.requests]
        inputs = [r.body.get("input") for r in server.requests]

    assert result == "done"
    assert seen == [_OPENAI_PRIMARY, _OPENAI_PRIMARY, _ANTHROPIC_OTHER]
    assert efforts[:2] == ["high", "low"]
    # The retry re-sent the same request: nothing was appended for the cut-off turn.
    assert inputs[0] == inputs[1]
    assert printer.notices() == [
        "Retrying the request at reasoning effort low after max_output_tokens",
        f"Switched model from {_OPENAI_PRIMARY} to {_ANTHROPIC_OTHER} after an output "
        'cap cut-off (stop_reason="max_output_tokens")',
    ]
    assert [c["stop_reason"] for c in printer.llm_calls()[:2]] == ["max_output_tokens"] * 2


def test_output_cap_retry_that_succeeds_needs_no_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the low-effort retry completes, the run stays on its model."""
    _register(monkeypatch, _OPENAI_PRIMARY, _ANTHROPIC_OTHER, thinking="high")
    _register(monkeypatch, _ANTHROPIC_OTHER, None)
    count = {"primary": 0}

    def respond(request: Request) -> Reply:
        model = str(request.body.get("model"))
        count["primary"] += 1
        if count["primary"] == 1:
            return _responses_incomplete("max_output_tokens", model)
        return _responses_finish(model)

    printer = CapturePrinter()
    with ScriptedOpenAIServer(respond) as server:
        result = _run_openai(server, printer, _OPENAI_PRIMARY, responses_api=True)
        seen = _models_seen(server)
        efforts = [(r.body.get("reasoning") or {}).get("effort") for r in server.requests]

    assert result == "done"
    assert seen == [_OPENAI_PRIMARY, _OPENAI_PRIMARY]
    assert efforts == ["high", "low"]
    assert printer.notices() == [
        "Retrying the request at reasoning effort low after max_output_tokens"
    ]


@pytest.mark.parametrize("catalog", [{}, {"thinking": "low"}], ids=["no-knob", "already-low"])
def test_model_without_a_lower_effort_switches_without_a_retry(
    monkeypatch: pytest.MonkeyPatch, catalog: dict[str, Any]
) -> None:
    """No catalog ``thinking`` level, or one already at ``low``, leaves no
    lower effort to retry at."""
    _register(monkeypatch, _OPENAI_PRIMARY, _ANTHROPIC_OTHER, **catalog)
    _register(monkeypatch, _ANTHROPIC_OTHER, None)

    def respond(request: Request) -> Reply:
        model = str(request.body.get("model"))
        if model == _OPENAI_PRIMARY:
            return _responses_incomplete("max_output_tokens", model)
        return _responses_finish(model)

    printer = CapturePrinter()
    with ScriptedOpenAIServer(respond) as server:
        result = _run_openai(server, printer, _OPENAI_PRIMARY, responses_api=True)
        seen = _models_seen(server)

    assert result == "done"
    assert seen == [_OPENAI_PRIMARY, _ANTHROPIC_OTHER]
    assert len(printer.notices()) == 1 and printer.notices()[0].startswith("Switched model")


def test_same_family_fallback_is_announced_when_nothing_else_is_configured(
    monkeypatch: pytest.MonkeyPatch, no_other_family: None
) -> None:
    """With no other family configured the declared same-family fallback is
    the last resort, and the notice says so."""
    _register(monkeypatch, _OPENAI_PRIMARY, _OPENAI_TWIN, thinking="high")
    _register(monkeypatch, _OPENAI_TWIN, None)

    def respond(request: Request) -> Reply:
        model = str(request.body.get("model"))
        if model == _OPENAI_PRIMARY:
            return _responses_incomplete("content_filter", model)
        return _responses_finish(model)

    printer = CapturePrinter()
    with ScriptedOpenAIServer(respond) as server:
        result = _run_openai(server, printer, _OPENAI_PRIMARY, responses_api=True)
        seen = _models_seen(server)

    assert result == "done"
    assert seen == [_OPENAI_PRIMARY, _OPENAI_TWIN]
    assert printer.notices() == [
        f"Switched model from {_OPENAI_PRIMARY} to {_OPENAI_TWIN} after a content filter "
        'stop (stop_reason="content_filter"); no other model family is configured'
    ]


def test_content_filter_with_no_fallback_at_all_ends_the_run(
    monkeypatch: pytest.MonkeyPatch, no_other_family: None
) -> None:
    """Nothing to switch to: the typed error reaches the caller after the one
    context-stating retry a refusal gets, and every call was billed."""
    _register(monkeypatch, _OPENAI_PRIMARY, None, thinking="high")

    def respond(request: Request) -> Reply:
        return _responses_incomplete("content_filter", str(request.body.get("model")))

    printer = CapturePrinter()
    with ScriptedOpenAIServer(respond) as server:
        with pytest.raises(ContentFilterError, match="content_filter"):
            _run_openai(server, printer, _OPENAI_PRIMARY, responses_api=True)
        seen = _models_seen(server)

    assert seen == [_OPENAI_PRIMARY, _OPENAI_PRIMARY]
    assert printer.notices() == []
    assert [c["stop_reason"] for c in printer.llm_calls()] == ["content_filter"] * 2


@pytest.mark.parametrize(
    ("reason", "error_type"),
    [
        ("content_filter", ContentFilterError),
        ("max_output_tokens", OutputCapError),
        ("unknown_reason", KISSError),
    ],
)
def test_streamed_responses_incomplete_is_typed_by_its_reason(
    reason: str, error_type: type[KISSError]
) -> None:
    """The streaming terminal event is typed like the blocking status; a reason
    the adapter does not know stays a plain error."""
    body = _responses_incomplete(reason, _OPENAI_PRIMARY).json_body

    def respond(request: Request) -> Reply:
        return Reply(sse_chunks=[responses_event("response.incomplete", {"response": body})])

    with ScriptedOpenAIServer(respond) as server:
        m = OpenAICompatibleModel2(
            _OPENAI_PRIMARY, base_url=server.base_url, api_key="sk-test", token_callback=print
        )
        m.conversation = [{"role": "user", "content": "hi"}]
        with pytest.raises(error_type) as excinfo:
            m.generate_and_process_with_tools({}, tools_schema=[_FINISH_TOOL])

    assert type(excinfo.value) is error_type
    assert reason in str(excinfo.value)
    billed = m.take_partial_usage_response()
    assert m.extract_input_output_token_counts_from_response(billed)[:2] == (700, 300)


# --- Chat Completions: typed finish reasons -------------------------------------


@pytest.mark.parametrize(
    ("finish_reason", "error_type"),
    [("content_filter", ContentFilterError), ("length", OutputCapError)],
)
def test_chat_completions_types_the_cut_short_finish_reason(
    finish_reason: str, error_type: type[ContentFilterError | OutputCapError]
) -> None:
    """The adapter raises the typed error, naming the finish reason, and keeps
    the billed response for the agent's accounting."""

    def respond(request: Request) -> Reply:
        return _chat_cut_short(finish_reason, str(request.body.get("model")))

    with ScriptedOpenAIServer(respond) as server:
        m = OpenAICompatibleModel(_OPENAI_PRIMARY, base_url=server.base_url, api_key="sk-test")
        m.conversation = [{"role": "user", "content": "hi"}]
        with pytest.raises(error_type) as excinfo:
            m.generate_and_process_with_tools({}, tools_schema=[_FINISH_TOOL])

    error = excinfo.value
    assert isinstance(error, ContentFilterError | OutputCapError)
    assert error.stop_reason == finish_reason
    assert finish_reason in str(error)
    billed = m.take_partial_usage_response()
    assert m.extract_input_output_token_counts_from_response(billed)[:2] == (700, 300)


def test_streamed_length_cut_off_bills_with_its_stop_reason(
    monkeypatch: pytest.MonkeyPatch, no_other_family: None
) -> None:
    """A streamed ``length`` cut-off is an ``OutputCapError``; the usage chunk
    carries no finish reason, so the error's own names the ``llm_call`` record."""
    _register(monkeypatch, _OPENAI_PRIMARY, None)

    def respond(request: Request) -> Reply:
        return _chat_streamed_length(str(request.body.get("model")))

    printer = CapturePrinter(streaming=True)
    with ScriptedOpenAIServer(respond) as server:
        with pytest.raises(OutputCapError, match="length"):
            _run_openai(server, printer, _OPENAI_PRIMARY, responses_api=False)

    calls = printer.llm_calls()
    assert len(calls) == 1
    assert calls[0]["stop_reason"] == "length"
    assert calls[0]["input_tokens"] == 700 and calls[0]["output_tokens"] == 300


# --- Anthropic: max_tokens and refusal -------------------------------------------


def _sse(event_type: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


def _anthropic_stream(
    model: str,
    stop_reason: str,
    blocks: list[tuple[dict[str, Any], str]],
    output_tokens: int = 50,
) -> list[bytes]:
    """An Anthropic Messages stream of *blocks* ending with *stop_reason*.

    Args:
        model: The model name echoed back.
        stop_reason: The final ``stop_reason``.
        blocks: ``(content_block_start payload, delta text)`` pairs: a
            ``text`` block's text, or a ``tool_use`` block's (possibly
            cut off) partial JSON.
        output_tokens: The billed output size.
    """
    events = [
        _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 11, "output_tokens": 1},
                },
            },
        )
    ]
    for index, (block, delta) in enumerate(blocks):
        events.append(
            _sse(
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": block},
            )
        )
        if block["type"] == "tool_use":
            payload = {"type": "input_json_delta", "partial_json": delta}
        else:
            payload = {"type": "text_delta", "text": delta}
        events.append(
            _sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": index, "delta": payload},
            )
        )
        events.append(_sse("content_block_stop", {"type": "content_block_stop", "index": index}))
    events.append(
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        )
    )
    events.append(_sse("message_stop", {"type": "message_stop"}))
    return events


def _tool_block(call_id: str, name: str) -> dict[str, Any]:
    return {"type": "tool_use", "id": call_id, "name": name, "input": {}}


_TEXT_BLOCK: dict[str, Any] = {"type": "text", "text": ""}

_ANTHROPIC_SCRIPTS: dict[str, list[list[bytes]]] = {}
"""Per model name, the streams to serve in order (the last one repeats)."""
_ANTHROPIC_SEEN: list[str] = []
_ANTHROPIC_LOCK = threading.Lock()


class _AnthropicHandler(BaseHTTPRequestHandler):
    """Serves each model's scripted streams in order."""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        model = str(body.get("model", ""))
        with _ANTHROPIC_LOCK:
            _ANTHROPIC_SEEN.append(model)
            scripts = _ANTHROPIC_SCRIPTS[model]
            chunks = scripts.pop(0) if len(scripts) > 1 else scripts[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(chunk)
            self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture(scope="module")
def anthropic_server() -> Generator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AnthropicHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture
def anthropic_scripts() -> Iterator[dict[str, list[list[bytes]]]]:
    _ANTHROPIC_SCRIPTS.clear()
    _ANTHROPIC_SEEN.clear()
    yield _ANTHROPIC_SCRIPTS
    _ANTHROPIC_SCRIPTS.clear()


def _anthropic_model(url: str, name: str = _ANTHROPIC_PRIMARY) -> AnthropicModel:
    m = AnthropicModel(name, api_key="test-key")
    m.client = anthropic.Anthropic(api_key="test-key", base_url=url)
    m.conversation = [{"role": "user", "content": "Do the task."}]
    return m


def test_anthropic_max_tokens_inside_the_only_tool_call_raises_output_cap(
    anthropic_server: str, anthropic_scripts: dict[str, list[list[bytes]]]
) -> None:
    """A ``tool_use`` block the cap cut off is not a tool call: the adapter
    raises instead of returning a text turn, keeps the billed usage for the
    agent and leaves the conversation as it was."""
    anthropic_scripts[_ANTHROPIC_PRIMARY] = [
        _anthropic_stream(
            _ANTHROPIC_PRIMARY,
            "max_tokens",
            [(_TEXT_BLOCK, "Let me "), (_tool_block("toolu_1", "finish"), '{"result": "do')],
            output_tokens=64,
        )
    ]
    m = _anthropic_model(anthropic_server)
    before = [dict(msg) for msg in m.conversation]
    with pytest.raises(OutputCapError) as excinfo:
        m.generate_and_process_with_tools({}, tools_schema=[_FINISH_TOOL])
    assert excinfo.value.stop_reason == "max_tokens"
    assert 'stop_reason="max_tokens"' in str(excinfo.value)
    assert m.conversation == before
    billed = m.take_partial_usage_response()
    assert m.extract_input_output_token_counts_from_response(billed)[:2] == (11, 64)
    assert m.take_partial_usage_response() is None


def test_anthropic_max_tokens_in_text_only_raises_output_cap(
    anthropic_server: str, anthropic_scripts: dict[str, list[list[bytes]]]
) -> None:
    anthropic_scripts[_ANTHROPIC_PRIMARY] = [
        _anthropic_stream(_ANTHROPIC_PRIMARY, "max_tokens", [(_TEXT_BLOCK, "Thinking out loud")])
    ]
    m = _anthropic_model(anthropic_server)
    with pytest.raises(OutputCapError):
        m.generate_and_process_with_tools({}, tools_schema=[_FINISH_TOOL])


def test_anthropic_max_tokens_on_a_tool_less_turn_returns_the_cut_off_text(
    anthropic_server: str, anthropic_scripts: dict[str, list[list[bytes]]]
) -> None:
    """A caller that asked for text (a reviewer, a summarizer) can still read a
    cut-off answer; only a tool-bearing turn with no tool call left raises."""
    anthropic_scripts[_ANTHROPIC_PRIMARY] = [
        _anthropic_stream(_ANTHROPIC_PRIMARY, "max_tokens", [(_TEXT_BLOCK, "The review so far")])
    ]
    m = _anthropic_model(anthropic_server)
    text, _response = m.generate()
    assert text == "The review so far"
    assert m.conversation[-1]["role"] == "assistant"


def test_anthropic_max_tokens_keeps_the_tool_calls_completed_before_the_cut(
    anthropic_server: str, anthropic_scripts: dict[str, list[list[bytes]]]
) -> None:
    """Only the last block can be cut off; the tool calls before it were
    completed and are returned."""
    anthropic_scripts[_ANTHROPIC_PRIMARY] = [
        _anthropic_stream(
            _ANTHROPIC_PRIMARY,
            "max_tokens",
            [
                (_tool_block("toolu_1", "finish"), '{"result": "first"}'),
                (_tool_block("toolu_2", "finish"), '{"result": "cut o'),
            ],
        )
    ]
    m = _anthropic_model(anthropic_server)
    calls, _text, _response = m.generate_and_process_with_tools({}, tools_schema=[_FINISH_TOOL])
    assert [(c["id"], c["arguments"]) for c in calls] == [("toolu_1", {"result": "first"})]


def _refusal_stream(model: str) -> list[bytes]:
    return _anthropic_stream(model, "refusal", [], output_tokens=9)


def _point_at(anthropic_url: str, openai_url: str) -> Any:
    """A ``pre_step_hook`` that routes each adapter to its local server."""

    def hook(m: Any) -> None:
        if isinstance(m, AnthropicModel):
            if anthropic_url not in str(getattr(m.client, "base_url", "")):
                m.client = anthropic.Anthropic(api_key="test-key", base_url=anthropic_url)
        elif isinstance(m, OpenAICompatibleBase):
            m.base_url = openai_url
            m.api_key = "sk-test"

    return hook


def test_anthropic_refusal_fails_over_to_an_openai_family_model(
    monkeypatch: pytest.MonkeyPatch,
    anthropic_server: str,
    anthropic_scripts: dict[str, list[list[bytes]]],
) -> None:
    """A Claude refusal continues on a GPT-family model over the other wire
    format: the conversation is handed across and the fallback finishes."""
    _register(monkeypatch, _ANTHROPIC_PRIMARY, _OPENAI_OTHER, extended_thinking=False)
    _register(monkeypatch, _OPENAI_OTHER, None)
    _ensure_key(monkeypatch, "ANTHROPIC_API_KEY")
    _ensure_key(monkeypatch, "OPENAI_API_KEY")
    anthropic_scripts[_ANTHROPIC_PRIMARY] = [_refusal_stream(_ANTHROPIC_PRIMARY)]

    def respond(request: Request) -> Reply:
        assert request.path.endswith("/chat/completions")
        return _chat_finish(str(request.body.get("model")))

    printer = CapturePrinter()
    with ScriptedOpenAIServer(respond) as server:
        agent = KISSAgent("refusal-crosses-families")
        agent.pre_step_hook = _point_at(anthropic_server, server.base_url)
        result = agent.run(
            model_name=_ANTHROPIC_PRIMARY,
            prompt_template="Do the task.",
            max_steps=5,
            max_budget=1.0,
            verbose=False,
            printer=printer,
        )
        openai_seen = _models_seen(server)
        openai_messages = server.requests[0].body["messages"]

    assert result == "done"
    assert agent.model_name == _OPENAI_OTHER
    assert _ANTHROPIC_SEEN == [_ANTHROPIC_PRIMARY]
    assert openai_seen == [_OPENAI_OTHER]
    # The task went across with the conversation.
    assert any("Do the task." in str(msg.get("content")) for msg in openai_messages)
    assert printer.notices() == [
        f"Switched model from {_ANTHROPIC_PRIMARY} to {_OPENAI_OTHER} after a safety "
        'refusal (stop_reason="refusal")'
    ]
    assert printer.llm_calls()[0]["stop_reason"] == "refusal"
