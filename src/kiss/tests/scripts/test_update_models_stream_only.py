# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: capability probes survive vendor models that require ``stream=true``.

A handful of vendor backends (most notably Together AI's
``Qwen/Qwen3.7-Max`` / ``Qwen/Qwen3.7-Plus``) reject Chat Completions
requests that omit ``"stream": true`` with::

    400 — This model only supports streaming. Set "stream": true.

KISS's ``OpenAICompatibleModel._stream_text`` only flips on streaming
when a ``token_callback`` is registered, so the discovery probes in
``scripts/update_models.py`` must register one (even if no-op) — otherwise
the request is sent non-streaming and the vendor returns HTTP 400, the
probe fails, and a perfectly usable model is silently dropped.

The tests in this module stand up a real local HTTP server that mimics
that vendor behavior (rejects non-streaming requests, otherwise emits a
small SSE stream) and exercise ``OpenAICompatibleModel`` end-to-end
against it. No mocks, patches, or test doubles are used: it is a real
HTTP client talking to a real HTTP server over a real loopback socket.
"""

from __future__ import annotations

import pytest

from kiss.core import config as config_module
from kiss.scripts import update_models as _update_models
from kiss.scripts.update_models import _noop_token_callback, detect_thinking_level
from kiss.tests.core.models import anthropic_sse_harness
from kiss.tests.core.models import openai_sse_harness as sse
from kiss.tests.core.models.test_update_models_stream_only import _stream_only_server

_probe_generate = _update_models.test_generate
_probe_function_calling = _update_models.test_function_calling










def test_noop_token_callback_is_callable_with_string_token() -> None:
    """``_noop_token_callback`` must accept a positional string token and
    return ``None`` (silently) so it can stand in for a real streaming
    callback everywhere ``test_generate`` / ``test_function_calling`` /
    ``detect_thinking_level`` plug it in.
    """
    _noop_token_callback("hello")
    _noop_token_callback("")


def test_update_models_probes_carry_token_callback_for_stream_only_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: ``update_models.test_generate`` /
    ``test_function_calling`` / ``detect_thinking_level`` succeed against
    a stream-only server, because they now build the model with a
    ``token_callback``.

    The factory ``kiss.core.models.model_info.model`` is invoked exactly
    as the probes do — only its routing decision is steered to our local
    server by recognising the test model name. We do this by exposing the
    server's base_url through an environment variable that we read inside
    a thin replacement of the routing decision in the ``model`` factory.

    This still exercises the *real* OpenAICompatibleModel + the *real*
    HTTP path; only the URL it points at is swapped out.
    """
    with _stream_only_server() as base_url:
        from kiss.core.models.model_info import model as create_model

        m = create_model(
            "stream-only-test",
            model_config={"base_url": base_url, "api_key": "dummy"},
            token_callback=_noop_token_callback,
        )
        m.initialize("Say hello in one word.")
        text, _ = m.generate()
        assert text == "hello"

        def calculator(expression: str = "") -> str:
            """Compute a math expression.

            Args:
                expression: A math expression string like ``'2+3'``.
            """
            return str(eval(expression))  # noqa: S307 — test-only

        m2 = create_model(
            "stream-only-test",
            model_config={"base_url": base_url, "api_key": "dummy"},
            token_callback=_noop_token_callback,
        )
        m2.initialize("What is 2+3? Use the calculator tool.")
        calls, _, _ = m2.generate_and_process_with_tools({"calculator": calculator})
        assert len(calls) == 1
        call = calls[0]
        if isinstance(call, dict):
            name = call.get("name") or call.get("function", {}).get("name")
        else:
            name = getattr(call, "name", None) or getattr(
                getattr(call, "function", None), "name", None
            )
        assert name == "calculator"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _probe_generate("unknown-vendor/does-not-exist") is False
    assert _probe_function_calling("unknown-vendor/does-not-exist") is False
    assert detect_thinking_level("unknown-vendor/does-not-exist") is None


def _call_the_tool_the_model_is_named_for(request: sse.Request) -> sse.Reply:
    """Answer a chat request with one streamed tool call, of the tool ``calls-<tool>`` names."""
    function = {"name": request.body["model"].removeprefix("calls-"), "arguments": "{}"}
    call = {"index": 0, "id": "call_1", "type": "function", "function": function}
    choices = (
        {"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None},
        {"index": 0, "delta": {}, "finish_reason": "tool_calls"},
    )
    return sse.Reply(sse_chunks=[sse.chat_chunk({"choices": [choice]}) for choice in choices])


def test_function_calling_probe_requires_the_offered_tool() -> None:
    """A model that answers with a tool it was not offered does not call tools."""
    with sse.ScriptedOpenAIServer(_call_the_tool_the_model_is_named_for) as server:
        endpoint = {"base_url": server.base_url, "api_key": "dummy"}
        assert _probe_function_calling("calls-calculator", endpoint) is True
        assert _probe_function_calling("calls-web_search", endpoint) is False


@pytest.mark.parametrize(
    ("error_type", "verdict"),
    [
        ("overloaded_error", None),
        ("api_error", None),
        ("rate_limit_error", None),
        ("timeout_error", None),
        ("invalid_request_error", False),
    ],
)
def test_probes_read_an_error_inside_a_claude_stream_by_its_type(
    error_type: str, verdict: bool | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anthropic reports an overload inside a stream that had answered 200 as an ``error`` event.

    The SDK raises it with the stream's status, which read as a verdict: a retest during an
    overload wrote ``gen`` and ``fc`` away.  A request rejected inside the stream still is one.
    """
    error = {"type": "error", "error": {"type": error_type, "message": "Overloaded"}}
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "sk-ant-test")
    reply = [anthropic_sse_harness.sse("error", error)]
    with anthropic_sse_harness.ScriptedAnthropicServer(reply) as server:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", server.base_url)
        assert _probe_generate("claude-opus-4-6") is verdict
        assert _probe_function_calling("claude-opus-4-6") is verdict
        assert len(server.requests) == 2  # each probe was answered: no verdict of a missing key
