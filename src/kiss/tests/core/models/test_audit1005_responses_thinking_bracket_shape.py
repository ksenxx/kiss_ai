# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: the Responses stream emits one balanced thinking bracket per reasoning run.

``OpenAICompatibleModel2._consume_stream_events`` used to track the
bracket in a local ``in_reasoning`` flag that mirrored
:attr:`Model._thinking_open` step for step (plus an ``opened_here``
dance for a ``reasoning_*_text.done`` suffix).  Both now go through the
shared ``_open_thinking_if_closed`` / ``_close_thinking_if_open``
helpers.  These tests pin the observable callback sequence on the
reasoning paths the flag used to guard (delta, ``.done`` with and
without a prior delta, text delta, ``response.completed``), over a real
``ThreadingHTTPServer`` speaking
genuine Responses-API SSE to the real ``openai`` SDK — no mocks or
patches.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    responses_event,
)

_MODEL = "gpt-responses-bracket-shape-under-test"


def _completed(output: list[dict[str, Any]]) -> bytes:
    """Render the terminal ``response.completed`` event for *output*."""
    return responses_event(
        "response.completed",
        {
            "response": {
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "model": _MODEL,
                "status": "completed",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "output": output,
                "usage": {
                    "input_tokens": 5,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 4,
                    "output_tokens_details": {"reasoning_tokens": 3},
                    "total_tokens": 9,
                },
            }
        },
    )


def _reasoning_delta(delta: str, seq: int) -> bytes:
    """One ``response.reasoning_summary_text.delta`` for summary 0 of item 0."""
    return responses_event(
        "response.reasoning_summary_text.delta",
        {
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "delta": delta,
            "sequence_number": seq,
        },
    )


def _reasoning_done(text: str, seq: int) -> bytes:
    """The ``response.reasoning_summary_text.done`` for summary 0 of item 0."""
    return responses_event(
        "response.reasoning_summary_text.done",
        {
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "text": text,
            "sequence_number": seq,
        },
    )


def _text_delta(delta: str, seq: int) -> bytes:
    """One ``response.output_text.delta`` for the message at output index 1."""
    return responses_event(
        "response.output_text.delta",
        {
            "item_id": "msg_1",
            "output_index": 1,
            "content_index": 0,
            "delta": delta,
            "sequence_number": seq,
        },
    )


_MESSAGE_OUTPUT = [
    {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "42", "annotations": []}],
    }
]

# Several reasoning deltas, a ``.done`` whose text extends them by a
# suffix, then the answer: ONE bracket around all the reasoning.
_DELTAS_THEN_DONE_WITH_SUFFIX = [
    _reasoning_delta("Let me ", 1),
    _reasoning_delta("think", 2),
    _reasoning_done("Let me think harder", 3),
    _text_delta("42", 4),
    _completed(_MESSAGE_OUTPUT),
]

# No reasoning delta at all: the ``.done`` alone carries the summary.
_DONE_ONLY = [
    _reasoning_done("Whole summary at once", 1),
    _text_delta("42", 2),
    _completed(_MESSAGE_OUTPUT),
]

# Reasoning that ends with the answer and no ``.done`` event: the text
# delta closes the bracket.
_DELTAS_THEN_TEXT = [
    _reasoning_delta("Pondering", 1),
    _text_delta("42", 2),
    _completed(_MESSAGE_OUTPUT),
]

# Reasoning still open when the stream completes: the post-loop close.
_DELTAS_THEN_COMPLETED = [
    _reasoning_delta("Pondering", 1),
    _completed(_MESSAGE_OUTPUT),
]

_SCRIPTS: dict[str, list[bytes]] = {}


def _responder(request: Request) -> Reply:
    """Answer with the script the request's ``user`` field names."""
    return Reply(sse_chunks=_SCRIPTS[request.body["user"]])


@pytest.fixture(scope="module")
def server() -> Generator[ScriptedOpenAIServer]:
    """A real Responses endpoint that replays the named script."""
    with ScriptedOpenAIServer(_responder) as srv:
        yield srv


def _run(server: ScriptedOpenAIServer, script_name: str) -> tuple[list[Any], str]:
    """Stream *script_name* through a real model, recording every callback.

    Returns:
        The interleaved callback log (``True``/``False`` for the thinking
        bracket, strings for tokens) and the returned content.
    """
    log: list[Any] = []
    model = OpenAICompatibleModel2(
        _MODEL,
        base_url=server.base_url,
        api_key="test-key",
        # ``user`` is forwarded verbatim, which is how the server picks a script.
        model_config={"stream_stall_timeout": 10.0, "user": script_name},
        token_callback=log.append,
        thinking_callback=log.append,
    )
    model.initialize("Think, then answer.")
    content, _response = model.generate()
    assert model._thinking_open is False
    return log, content


def test_done_suffix_extends_the_open_bracket(server: ScriptedOpenAIServer) -> None:
    """Deltas open the bracket once; the ``.done`` suffix streams inside it; one close."""
    _SCRIPTS["suffix"] = _DELTAS_THEN_DONE_WITH_SUFFIX
    log, content = _run(server, "suffix")
    assert log == [True, "Let me ", "think", " harder", False, "42"]
    assert content == "42"


def test_done_without_deltas_is_a_self_contained_bracket(
    server: ScriptedOpenAIServer,
) -> None:
    """A ``.done`` with no prior delta opens, streams the summary, and closes."""
    _SCRIPTS["done-only"] = _DONE_ONLY
    log, content = _run(server, "done-only")
    assert log == [True, "Whole summary at once", False, "42"]
    assert content == "42"


def test_text_delta_closes_the_bracket(server: ScriptedOpenAIServer) -> None:
    """Answer text arriving after reasoning closes the bracket first."""
    _SCRIPTS["text"] = _DELTAS_THEN_TEXT
    log, content = _run(server, "text")
    assert log == [True, "Pondering", False, "42"]
    assert content == "42"


def test_completed_closes_a_still_open_bracket(server: ScriptedOpenAIServer) -> None:
    """A stream completing mid-reasoning still ends with a balanced bracket."""
    _SCRIPTS["completed"] = _DELTAS_THEN_COMPLETED
    log, content = _run(server, "completed")
    # The final message text is emitted from the completed payload after
    # the bracket is closed.
    assert log == [True, "Pondering", False, "42"]
    assert content == "42"
