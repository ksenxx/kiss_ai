# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests locking down offline-testable behavior of
``openai_compatible_model.py`` (v1) and ``openai_compatible_model2.py`` (v2)
before/after code simplification.

No mocks, patches, fakes, or monkeypatching: every test calls real
functions and constructs real objects (``types.SimpleNamespace`` instances
are plain Python objects standing in for SDK response payloads; dict-shaped
responses are natively supported by the code under test).  No network calls
are made (``base_url="http://localhost:1"`` is never contacted).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.openai_compatible_model import (
    OpenAICompatibleModel,
)
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.test_simplify_models_regr_oai import (  # noqa: F401
    API_KEY,
    BASE_URL,
    MODEL,
    _dict_response,
    _fc,
    _fco,
    make_v2,
)


def test_finalize_stream_response() -> None:
    usage_chunk = SimpleNamespace(usage=object())
    last = SimpleNamespace(usage=None)
    assert OpenAICompatibleModel._finalize_stream_response(usage_chunk, last) is usage_chunk
    assert OpenAICompatibleModel._finalize_stream_response(None, last) is last
    with pytest.raises(KISSError):
        OpenAICompatibleModel._finalize_stream_response(None, None)


def test_shape_responses_kwargs() -> None:
    cfg = {
        "system_instruction": "sys",
        "reasoning_effort": "high",
        "max_tokens": 5,
        "stop": ["x"],
        "seed": 1,
        "temperature": 0.5,
        "response_format": {"type": "json_object"},
        "tool_choice": {"type": "function", "function": {"name": "f"}},
        "parallel_tool_calls": True,
        "enable_cache": True,
    }
    m = make_v2(model_config=dict(cfg))
    msgs = [{"role": "user", "content": "hi"}]
    kwargs = m._shape_responses_kwargs(input_items=msgs, tools=None)
    assert kwargs["model"] == MODEL
    assert kwargs["input"] == msgs
    assert kwargs["instructions"] == "sys"
    assert kwargs["reasoning"] == {"effort": "high", "summary": "auto"}
    assert kwargs["max_output_tokens"] == 5
    assert kwargs["text"] == {"format": {"type": "json_object"}}
    assert kwargs["temperature"] == 0.5
    for absent in (
        "stop",
        "seed",
        "system_instruction",
        "reasoning_effort",
        "response_format",
        "enable_cache",
        "tool_choice",
        "parallel_tool_calls",
        "tools",
        "max_tokens",
    ):
        assert absent not in kwargs

    tools = [{"type": "function", "name": "f", "parameters": {}}]
    kwargs2 = m._shape_responses_kwargs(input_items=msgs, tools=tools)
    assert kwargs2["tools"] == tools
    assert kwargs2["tool_choice"] == {"type": "function", "name": "f"}
    assert kwargs2["parallel_tool_calls"] is True

    m2 = make_v2(model_config={"max_tokens": 5, "max_completion_tokens": 9})
    assert m2._shape_responses_kwargs(input_items=msgs, tools=None)["max_output_tokens"] == 9

    with pytest.raises(KISSError):
        m._shape_responses_kwargs(input_items=[{"role": "user", "content": " "}], tools=None)

    m3 = make_v2(model_config={"reasoning": {"summary": "auto"}, "reasoning_effort": "low"})
    out3 = m3._shape_responses_kwargs(input_items=msgs, tools=None)
    assert out3["reasoning"] == {"summary": "auto", "effort": "low"}
    assert m3.model_config["reasoning"] == {"summary": "auto"}


def test_add_function_results_mismatch_rolls_back() -> None:
    m = make_v2()
    m.initialize("hi")
    m.conversation.append(_fc("c1"))
    m._pending_function_calls = [{"name": "f", "call_id": "c1"}]
    before = list(m.conversation)
    with pytest.raises(KISSError):
        m.add_function_results_to_conversation_and_return([("WRONG", {"result": "x"})])
    assert m.conversation == before
    assert m._pending_function_calls == [{"name": "f", "call_id": "c1"}]


def test_validate_function_call_conversation() -> None:
    m = make_v2()
    m.initialize("hi")
    m.conversation.extend([_fc("c1"), _fco("c1"), _fc("c2")])
    assert m._validate_function_call_conversation() == ["c2"]

    m2 = make_v2()
    m2.initialize("hi")
    m2.conversation.extend([_fc("c1"), _fc("c1")])
    with pytest.raises(KISSError):
        m2._validate_function_call_conversation()

    m3 = make_v2()
    m3.initialize("hi")
    m3.conversation.append(_fco("nope"))
    with pytest.raises(KISSError):
        m3._validate_function_call_conversation()

    m4 = make_v2()
    m4.initialize("hi")
    m4.conversation.extend([_fc("c1"), {"role": "user", "content": "early"}])
    with pytest.raises(KISSError):
        m4._validate_function_call_conversation()


def test_ensure_no_pending_function_calls_raises() -> None:
    m = make_v2()
    m.initialize("hi")
    m.conversation.append(_fc("c1"))
    with pytest.raises(KISSError):
        m._build_request_kwargs(tools=None)


def test_response_has_message_text_and_failed() -> None:
    has = OpenAICompatibleModel2._response_has_message_text
    assert has(_dict_response(
        [{"type": "message", "content": [{"type": "output_text", "text": ""}]}]
    ))
    assert not has(_dict_response([{"type": "reasoning"}]))

    OpenAICompatibleModel2._raise_for_failed_response(_dict_response([]))
    with pytest.raises(KISSError, match="boom"):
        OpenAICompatibleModel2._raise_for_failed_response(
            {"status": "failed", "error": {"message": "boom"}}
        )
    with pytest.raises(KISSError, match="max_output_tokens"):
        OpenAICompatibleModel2._raise_for_failed_response(
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
        )


def test_consume_stream_missing_completed_raises() -> None:
    m = make_v2(token_callback=lambda _t: None)
    m.initialize("hi")
    events = [
        SimpleNamespace(
            type="response.output_text.delta", output_index=0, content_index=0, delta="x"
        )
    ]
    with pytest.raises(KISSError):
        m._consume_stream(events)


def test_consume_stream_failed_event_raises() -> None:
    m = make_v2(token_callback=lambda _t: None)
    m.initialize("hi")
    events = [
        SimpleNamespace(type="response.failed", response=None, error={"message": "kaboom"})
    ]
    with pytest.raises(KISSError, match="kaboom"):
        m._consume_stream(events)
