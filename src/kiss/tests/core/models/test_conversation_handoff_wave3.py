# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Both model-switch sites hand the live conversation over through ``hand_off_conversation``.

Responses API items belong to the endpoint that produced them, so a
switch to another endpoint (Together -> OpenAI) converts the
conversation to Chat Completions messages, while a switch on the same
endpoint keeps it verbatim: an OpenAI model keeps another OpenAI model's
reasoning items.  The sites are Sorcar's ``set_model`` tool and
``KISSAgent``'s fallback.  Local servers stand in for the endpoints;
nothing is mocked.
"""

from __future__ import annotations

import copy
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model import hand_off_conversation, responses_items_to_chat_messages
from kiss.core.models.model_info import MODEL_INFO
from kiss.core.models.model_info import model as model_factory
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.agents.sorcar.test_change_model import (  # noqa: F401  (autouse fixtures)
    _find_tool,
    _isolate_config,
    _isolate_sorcar_db,
    _make_agent,
)
from kiss.tests.core.models.test_v2_handoff_output_items_wave2 import (
    _TOGETHER_CALL,
    _TOGETHER_MESSAGE,
    _TOGETHER_REASONING,
    _Endpoint,
    _items,
    _response,
    _text_message,
    _together_tool_turn,
    reveal_secret,
)

_TOGETHER_TURN = [
    {"role": "user", "content": "Play the memory game."},
    _TOGETHER_REASONING,
    _TOGETHER_MESSAGE,
    _TOGETHER_CALL,
    {"type": "function_call_output", "call_id": "call_t1", "output": "SECRET-2"},
]


@pytest.fixture
def together() -> Iterator[_Endpoint]:
    """One live Responses endpoint."""
    server = _Endpoint()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def test_same_endpoint_handoff_keeps_reasoning_items(together: _Endpoint) -> None:
    """A second model on the same endpoint replays the first one's items verbatim."""
    old = _together_tool_turn(together)
    new = OpenAICompatibleModel2("openai/gpt-oss-20b", base_url=together.base_url, api_key="k")
    new.initialize("Play the memory game.")
    hand_off_conversation(old, new)
    assert new.conversation == old.conversation
    assert new.conversation is not old.conversation
    new.add_message_to_conversation("user", "Which secret did you learn?")
    together.replies = [_response([_text_message("msg_responses-3", "SECRET-2.")])]
    assert new.generate()[0] == "SECRET-2."

    body = together.bodies[-1]
    assert _items(body, "reasoning") == [_TOGETHER_REASONING]
    assert _items(body, "function_call") == [_TOGETHER_CALL]
    assert _TOGETHER_MESSAGE in body["input"]


def _together_agent() -> tuple[Any, Any]:
    """A Sorcar agent whose live model is a Together model mid-task, and its ``set_model`` tool."""
    agent, tools = _make_agent()
    agent.model = model_factory("openai/gpt-oss-120b", model_config={"api_key": "k"})
    agent.model_name = "openai/gpt-oss-120b"
    agent.model.initialize("")
    agent.model.conversation = copy.deepcopy(_TOGETHER_TURN)
    agent.model.usage_info_for_messages = "tokens=42"
    return agent, _find_tool(tools, "set_model")


def test_set_model_to_another_endpoint_converts_the_conversation() -> None:
    """Together -> OpenAI through ``set_model``: no reasoning item and no item id survive."""
    agent, set_model = _together_agent()
    assert agent.model.base_url == "https://api.together.xyz/v1"

    set_model("gpt-4o")

    assert agent.model.base_url == "https://api.openai.com/v1"
    assert agent.model.conversation == responses_items_to_chat_messages(_TOGETHER_TURN)
    assert [item["role"] for item in agent.model.conversation] == ["user", "assistant", "tool"]
    assert agent.model.conversation[1]["tool_calls"] == [
        {
            "id": "call_t1",
            "type": "function",
            "function": {"name": "reveal_secret", "arguments": '{"index": 2}'},
        }
    ]
    assert "reasoning" not in json.dumps(agent.model.conversation)
    assert agent.model.usage_info_for_messages == "tokens=42"


def test_set_model_on_the_same_endpoint_keeps_the_conversation() -> None:
    """Together -> Together through ``set_model``: the items stay exactly as they were."""
    agent, set_model = _together_agent()

    set_model("openai/gpt-oss-20b")

    assert agent.model.base_url == "https://api.together.xyz/v1"
    assert agent.model.conversation == _TOGETHER_TURN
    assert agent.model.usage_info_for_messages == "tokens=42"


class _FallbackEndpoint(ThreadingHTTPServer):
    """A Responses endpoint serving a primary model that fails on its second request."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _FallbackHandler)
        self.bodies: list[dict[str, Any]] = []


class _FallbackHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        endpoint = cast(_FallbackEndpoint, self.server)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        endpoint.bodies.append(body)
        status, reply = 200, _response([_OPENAI_REASONING, _OPENAI_CALL])
        if body["model"] == "gpt-4o":
            reply = _response([_FINISH_CALL])
        elif len(endpoint.bodies) > 1:
            status = 404
            reply = {"error": {"type": "not_found_error", "message": "gpt-5.5 is not available"}}
        payload = json.dumps(reply).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


_OPENAI_REASONING = {"id": "rs_1", "type": "reasoning", "summary": []}
_OPENAI_CALL = {
    "id": "fc_1",
    "type": "function_call",
    "call_id": "call_1",
    "name": "reveal_secret",
    "arguments": '{"index": 2}',
    "status": "completed",
}
_FINISH_CALL = {
    "id": "fc_2",
    "type": "function_call",
    "call_id": "call_2",
    "name": "finish",
    "arguments": '{"result": "SECRET-2"}',
    "status": "completed",
}


def test_fallback_on_the_same_endpoint_keeps_reasoning_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``KISSAgent``'s fallback replays the primary's reasoning and call items verbatim."""
    primary_info = copy.copy(MODEL_INFO["gpt-5.5"])
    primary_info.fallback = "gpt-4o"
    monkeypatch.setitem(MODEL_INFO, "gpt-5.5", primary_info)
    server = _FallbackEndpoint()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        agent = KISSAgent("handoff-fallback")
        result = agent.run(
            model_name="gpt-5.5",
            prompt_template="Reveal secret 2.",
            tools=[reveal_secret],
            max_steps=5,
            max_budget=1.0,
            verbose=False,
            model_config={
                "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                "api_key": "k",
                "use_responses_api": True,
            },
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result == "SECRET-2"
    assert agent.model_name == "gpt-4o"
    assert [body["model"] for body in server.bodies] == ["gpt-5.5", "gpt-5.5", "gpt-4o"]
    handed_over = server.bodies[-1]
    assert _items(handed_over, "reasoning") == [_OPENAI_REASONING]
    assert _items(handed_over, "function_call") == [_OPENAI_CALL]
    assert [item["call_id"] for item in _items(handed_over, "function_call_output")] == ["call_1"]
