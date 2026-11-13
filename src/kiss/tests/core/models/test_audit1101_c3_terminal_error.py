# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 (C3/F4): one terminal-error classification for both OpenAI transports.

``_raise_for_finish_reason`` (Chat Completions) and
``_terminal_response_error`` (Responses) each typed a cut-short
completion the same way -- the output-token budget ->
:class:`OutputCapError` carrying the provider's reason, the content
filter -> :class:`ContentFilterError`, anything else a plain
:class:`KISSError` -- in two copies that differed only in the budget
reason's spelling (``length`` / ``max_output_tokens``) and in the
Responses ``status == "incomplete"`` guard.  The copies were merged into
``openai_compatible_model._terminal_error``; this suite pins the exact
table both transports must keep: the exception type (``type`` equality,
not ``isinstance``), its ``stop_reason`` and the transport's own message.

Every case drives the real SDK against a real local endpoint, streamed
and not.  A duplication finding has no failing behaviour: this suite
passes before and after the merge by design.
"""

from __future__ import annotations

from typing import Any

import pytest

from kiss.core.kiss_error import ContentFilterError, KISSError, OutputCapError
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    chat_chunk,
    responses_event,
)

_MODEL = "gpt-audit1101-terminal"
_DETAIL = "upstream said no"
_CHAT_USAGE = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
_RESPONSES_USAGE = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}

# (finish_reason, expected type, expected stop_reason)
_CHAT_TABLE = [
    ("length", OutputCapError, "length"),
    ("content_filter", ContentFilterError, "content_filter"),
    ("error", KISSError, None),
]
# (status, incomplete reason, expected type, expected stop_reason)
_RESPONSES_TABLE = [
    ("incomplete", "max_output_tokens", OutputCapError, "max_output_tokens"),
    ("incomplete", "content_filter", ContentFilterError, "content_filter"),
    ("incomplete", "length", KISSError, None),
    ("incomplete", "", KISSError, None),
    ("failed", "content_filter", KISSError, None),
    ("failed", "max_output_tokens", KISSError, None),
    ("failed", "", KISSError, None),
]


def _chat_reply(request: Request, finish_reason: str) -> Reply:
    """A Chat Completions answer cut short with *finish_reason* and a gateway detail."""
    base = {"id": "chatcmpl-cut", "object": "chat.completion.chunk", "created": 0, "model": _MODEL}
    error = {"code": 502, "message": _DETAIL}
    if request.body.get("stream"):
        return Reply(
            sse_chunks=[
                chat_chunk(
                    {
                        **base,
                        "choices": [
                            {"index": 0, "delta": {"content": "part"}, "finish_reason": None}
                        ],
                    }
                ),
                chat_chunk(
                    {
                        **base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason,
                                "error": error,
                            }
                        ],
                    }
                ),
                chat_chunk({**base, "choices": [], "usage": _CHAT_USAGE}),
                b"data: [DONE]\n\n",
            ]
        )
    return Reply(
        json_body={
            **base,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "part"},
                    "finish_reason": finish_reason,
                    "error": error,
                }
            ],
            "usage": _CHAT_USAGE,
        }
    )


def _responses_reply(request: Request, status: str, reason: str) -> Reply:
    """A Responses answer ending in *status*; a failure carries the gateway detail."""
    body: dict[str, Any] = {
        "id": "resp_cut",
        "object": "response",
        "created_at": 0,
        "model": _MODEL,
        "status": status,
        "output": [],
        "usage": _RESPONSES_USAGE,
    }
    if reason:
        body["incomplete_details"] = {"reason": reason}
    if status == "failed":
        body["error"] = {"code": "server_error", "message": _DETAIL}
    if request.body.get("stream"):
        return Reply(sse_chunks=[responses_event(f"response.{status}", {"response": body})])
    return Reply(json_body=body)


def _assert_classified(
    error: KISSError, expected_type: type[KISSError], stop_reason: str | None
) -> None:
    assert type(error) is expected_type
    assert getattr(error, "stop_reason", None) == stop_reason


@pytest.mark.parametrize("streaming", [False, True], ids=["non-streamed", "streamed"])
@pytest.mark.parametrize(("finish_reason", "expected_type", "stop_reason"), _CHAT_TABLE)
def test_chat_completions_classification(
    streaming: bool, finish_reason: str, expected_type: type[KISSError], stop_reason: str | None
) -> None:
    """Chat Completions: ``length`` is the cap, ``content_filter`` the filter, else plain."""

    def respond(request: Request) -> Reply:
        return _chat_reply(request, finish_reason)

    with ScriptedOpenAIServer(respond) as server:
        model = OpenAICompatibleModel(
            _MODEL,
            base_url=server.base_url,
            api_key="k",
            token_callback=(lambda _t: None) if streaming else None,
        )
        model.initialize("hi")
        with pytest.raises(KISSError) as info:
            model.generate()
    _assert_classified(info.value, expected_type, stop_reason)
    message = str(info.value)
    assert f"finish_reason={finish_reason!r}" in message
    assert _DETAIL in message
    assert ("Raise max_tokens" in message) == (finish_reason == "length")
    rejected = model.take_partial_usage_response()
    assert model.extract_input_output_token_counts_from_response(rejected)[:2] == (7, 3)


@pytest.mark.parametrize("streaming", [False, True], ids=["non-streamed", "streamed"])
@pytest.mark.parametrize(("status", "reason", "expected_type", "stop_reason"), _RESPONSES_TABLE)
def test_responses_classification(
    streaming: bool,
    status: str,
    reason: str,
    expected_type: type[KISSError],
    stop_reason: str | None,
) -> None:
    """Responses: only an ``incomplete`` status is typed, and only by its two known reasons."""

    def respond(request: Request) -> Reply:
        return _responses_reply(request, status, reason)

    with ScriptedOpenAIServer(respond) as server:
        model = OpenAICompatibleModel2(
            _MODEL,
            base_url=server.base_url,
            api_key="k",
            token_callback=(lambda _t: None) if streaming else None,
        )
        model.initialize("hi")
        with pytest.raises(KISSError) as info:
            model.generate()
    _assert_classified(info.value, expected_type, stop_reason)
    message = str(info.value)
    if streaming:
        ending = "incomplete" if status == "incomplete" else "with response.failed"
        assert f"Responses API stream ended {ending}" in message
    else:
        assert f"Responses API returned {status} response" in message
    # The transport's own detail is kept: the failure message, else the reason.
    assert (_DETAIL if status == "failed" else reason) in message
    # So is the provider's code, which tells endpoint trouble from a refused request.
    assert getattr(info.value, "code", None) == ("server_error" if status == "failed" else None)
    rejected = model.take_partial_usage_response()
    assert model.extract_input_output_token_counts_from_response(rejected)[:2] == (7, 3)


def test_bare_error_event_carries_its_code() -> None:
    """The stream's ``error`` event has no response: its code was kept nowhere."""

    def respond(request: Request) -> Reply:
        error = {"code": "rate_limit_exceeded", "message": _DETAIL}
        return Reply(sse_chunks=[responses_event("error", error)])

    with ScriptedOpenAIServer(respond) as server:
        model = OpenAICompatibleModel2(
            _MODEL, base_url=server.base_url, api_key="k", token_callback=lambda _t: None
        )
        model.initialize("hi")
        with pytest.raises(KISSError) as info:
            model.generate()
    _assert_classified(info.value, KISSError, None)
    assert str(info.value) == f"KISS Error: Responses API stream ended with error: {_DETAIL}"
    assert getattr(info.value, "code", None) == "rate_limit_exceeded"


@pytest.mark.parametrize("status", ["queued", "in_progress", "cancelled"])
def test_unfinished_responses_reply_is_refused(status: str) -> None:
    """A non-streamed reply that never completed was returned as an empty answer."""

    def respond(request: Request) -> Reply:
        return _responses_reply(request, status, "")

    with ScriptedOpenAIServer(respond) as server:
        model = OpenAICompatibleModel2(_MODEL, base_url=server.base_url, api_key="k")
        model.initialize("hi")
        turns = len(model.conversation)
        with pytest.raises(KISSError) as info:
            model.generate()
    _assert_classified(info.value, KISSError, None)
    assert f"Responses API returned {status} response" in str(info.value)
    assert len(model.conversation) == turns
    rejected = model.take_partial_usage_response()
    assert model.extract_input_output_token_counts_from_response(rejected)[:2] == (7, 3)
