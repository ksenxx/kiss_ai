# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Responses conversation handed to another endpoint carries no foreign item state.

The live ``test_together_hop_via_set_model_tool`` failed with OpenAI's
400 ``Missing required parameter: 'input[8].summary'``: the
conversation handed back from a Together ``gpt-oss`` model (Responses
API) to ``gpt-4o`` still held Together's output items verbatim.  OpenAI
rejects all of their endpoint-bound state: the reasoning item has no
``summary`` and has a ``status``, and Together's ``function_call`` ids
(equal to the call ids, ``call_…``) are not OpenAI item ids (``fc_…``).
A hand-off between endpoints
(:func:`kiss.core.models.model.hand_off_conversation`) now converts the
conversation to Chat Completions messages, which keep no reasoning item
and no item id; the model that produced the items still replays them
verbatim.

Two local servers stand in for the two endpoints and record every
request body; nothing is mocked.
"""

from __future__ import annotations

import copy
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import pytest

from kiss.core.models.model import hand_off_conversation
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2

_USAGE = {
    "input_tokens": 5,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 4,
    "output_tokens_details": {"reasoning_tokens": 3},
    "total_tokens": 9,
}

# Output items in the shape Together's Responses endpoint returns them.
_TOGETHER_REASONING = {
    "id": "reasoning_responses-1",
    "type": "reasoning",
    "status": "completed",
    "content": [{"type": "reasoning_text", "text": "I should call reveal_secret."}],
}
_TOGETHER_MESSAGE = {
    "id": "msg_responses-1",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "Calling the tool.", "annotations": []}],
}
_TOGETHER_CALL = {
    "id": "call_t1",
    "type": "function_call",
    "call_id": "call_t1",
    "name": "reveal_secret",
    "arguments": '{"index": 2}',
    "status": "completed",
}


def _text_message(item_id: str, text: str) -> dict[str, Any]:
    """An assistant ``message`` output item carrying *text*."""
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _response(output: list[dict[str, Any]]) -> dict[str, Any]:
    """A completed non-streamed Responses body with *output*."""
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "m",
        "status": "completed",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": output,
        "usage": _USAGE,
    }


class _Endpoint(ThreadingHTTPServer):
    """A Responses endpoint answering with scripted bodies and recording each request."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.replies: list[dict[str, Any]] = []
        self.bodies: list[dict[str, Any]] = []

    @property
    def base_url(self) -> str:
        """The ``base_url`` a model pointed at this endpoint uses."""
        return f"http://127.0.0.1:{self.server_port}/v1"


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        endpoint = cast(_Endpoint, self.server)
        length = int(self.headers.get("Content-Length", 0))
        endpoint.bodies.append(json.loads(self.rfile.read(length)))
        payload = json.dumps(endpoint.replies.pop(0)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


@pytest.fixture
def endpoints() -> Iterator[tuple[_Endpoint, _Endpoint]]:
    """Two live endpoints: (together, openai)."""
    servers = (_Endpoint(), _Endpoint())
    for server in servers:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield servers
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


def reveal_secret(index: int) -> str:
    """Return the secret word with the given index.

    Args:
        index: The secret's index.
    """
    return f"SECRET-{index}"


def _items(body: dict[str, Any], item_type: str) -> list[dict[str, Any]]:
    """The ``input`` items of a request *body* with ``type`` *item_type*."""
    return [item for item in body["input"] if item.get("type") == item_type]


def _assistant_texts(body: dict[str, Any]) -> list[str]:
    """The text of every assistant message in a request *body*'s ``input``."""
    texts: list[str] = []
    for item in body["input"]:
        if item.get("role") == "assistant":
            content = item["content"]
            texts += [content] if isinstance(content, str) else [part["text"] for part in content]
    return texts


def _together_tool_turn(together: _Endpoint) -> OpenAICompatibleModel2:
    """Run one tool turn on a Together-served model, ending with its text answer."""
    together.replies = [
        _response(
            [
                copy.deepcopy(_TOGETHER_REASONING),
                copy.deepcopy(_TOGETHER_MESSAGE),
                copy.deepcopy(_TOGETHER_CALL),
            ]
        ),
        _response([_text_message("msg_responses-2", "The secret is SECRET-2.")]),
    ]
    model = OpenAICompatibleModel2("openai/gpt-oss-120b", base_url=together.base_url, api_key="k")
    model.initialize("Play the memory game.")
    calls, _, _ = model.generate_and_process_with_tools({"reveal_secret": reveal_secret})
    assert [call["name"] for call in calls] == ["reveal_secret"]
    model.add_function_results_to_conversation_and_return(
        [("reveal_secret", {"result": "SECRET-2"})]
    )
    assert model.generate()[0] == "The secret is SECRET-2."
    return model


def test_own_output_items_replay_verbatim(endpoints: tuple[_Endpoint, _Endpoint]) -> None:
    """The model that produced the items sends them back unchanged within its tool turn."""
    together, _ = endpoints
    _together_tool_turn(together)
    replay = together.bodies[1]
    assert _items(replay, "reasoning") == [_TOGETHER_REASONING]
    assert _items(replay, "function_call") == [_TOGETHER_CALL]
    assert _TOGETHER_MESSAGE in replay["input"]


def test_handoff_to_another_endpoint_drops_foreign_item_state(
    endpoints: tuple[_Endpoint, _Endpoint],
) -> None:
    """Together -> OpenAI: no reasoning item and no Together item id reaches OpenAI."""
    together, openai = endpoints
    old = _together_tool_turn(together)
    new = OpenAICompatibleModel2("gpt-5.5", base_url=openai.base_url, api_key="k")
    new.initialize("Play the memory game.")
    hand_off_conversation(old, new)
    new.add_message_to_conversation("user", "Which secret did you learn?")
    openai.replies = [_response([_text_message("msg_o1", "SECRET-2.")])]
    assert new.generate()[0] == "SECRET-2."

    body = openai.bodies[0]
    assert _items(body, "reasoning") == []
    assert _items(body, "function_call") == [
        {
            "type": "function_call",
            "call_id": "call_t1",
            "name": "reveal_secret",
            "arguments": '{"index": 2}',
        }
    ]
    assert _items(body, "function_call_output") == [
        {"type": "function_call_output", "call_id": "call_t1", "output": "SECRET-2"}
    ]
    assert all("id" not in item and "status" not in item for item in body["input"])
    assert _assistant_texts(body) == ["Calling the tool.", "The secret is SECRET-2."]
    # The converter copies: the old model's items are not edited.
    assert _TOGETHER_REASONING in old.conversation
    assert _TOGETHER_CALL in old.conversation


def test_items_that_are_not_dicts_pass_through(endpoints: tuple[_Endpoint, _Endpoint]) -> None:
    """The conversion leaves an item it cannot inspect where it is."""
    _, openai = endpoints
    model = OpenAICompatibleModel2("gpt-5.5", base_url=openai.base_url, api_key="k")
    model.initialize("Play the memory game.")
    opaque = object()
    assert model._foreign_items_to_native_input([opaque, *model.conversation]) == [
        opaque,
        *model.conversation,
    ]
