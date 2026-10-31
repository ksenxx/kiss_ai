# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: a billed response the failure checks refuse is recorded by the request path.

``_raise_for_failed_response`` (Responses) and ``_raise_for_failed_completion``
(Chat) used to store the response in ``_rejected_response`` themselves, a
second copy of what every caller had just stored: ``_blocking_request``
records any reply that arrived, inline or from its worker thread when a
stop event is bound, and the streamed consumers record their terminal
response.  The checks now only raise.  These tests pin that the usage of
a refused non-streamed response still reaches
``take_partial_usage_response`` through the request path, on both
transports and both ``_blocking_request`` paths.
"""

from __future__ import annotations

import threading

import pytest

from kiss.core import stop_signal
from kiss.core.kiss_error import KISSError
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import Reply, Request, ScriptedOpenAIServer

_MODEL = "gpt-failed-billing-under-test"


def _failed_reply(request: Request) -> Reply:
    """Answer with a billed failure: a ``failed`` response, or a Chat reply without choices."""
    if request.path.endswith("/responses"):
        return Reply(
            json_body={
                "id": "resp_failed",
                "object": "response",
                "created_at": 0,
                "model": _MODEL,
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "status": "failed",
                "error": {"code": "server_error", "message": "boom"},
                "output": [],
                "usage": {
                    "input_tokens": 700,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 300,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 1000,
                },
            }
        )
    return Reply(
        json_body={
            "id": "chatcmpl-failed",
            "object": "chat.completion",
            "created": 0,
            "model": _MODEL,
            "choices": [],
            "error": {"message": "boom"},
            "usage": {"prompt_tokens": 700, "completion_tokens": 300, "total_tokens": 1000},
        }
    )


@pytest.mark.parametrize("stoppable", [False, True], ids=["inline", "worker-thread"])
@pytest.mark.parametrize(
    "model_class", [OpenAICompatibleModel, OpenAICompatibleModel2], ids=["chat", "responses"]
)
def test_a_refused_response_is_billed_through_the_request_path(
    model_class: type, stoppable: bool
) -> None:
    with ScriptedOpenAIServer(_failed_reply) as server:
        model = model_class(_MODEL, base_url=server.base_url, api_key="sk-test")
        model.initialize("hi")
        stop_signal.set_thread_stop_event(threading.Event() if stoppable else None)
        try:
            with pytest.raises(KISSError, match="boom"):
                model.generate()
        finally:
            stop_signal.set_thread_stop_event(None)
    rejected = model.take_partial_usage_response()
    assert model.extract_input_output_token_counts_from_response(rejected)[:2] == (700, 300)
    assert model.take_partial_usage_response() is None
