"""Audit 2026-10-12 (core-models-b): Responses transport, end to end.

Every case drives a real ``OpenAICompatibleModel2`` (and, for parity, a
real ``OpenAICompatibleModel`` delegating to ``/v1/responses``) through
the real OpenAI SDK against a local ``ThreadingHTTPServer`` that speaks
the Responses protocol.  Nothing is seeded or patched: the function calls
come from the server, the results go through the public
``add_function_results_to_conversation_and_return`` and the evidence is
the request body the server received plus whether the next ``generate()``
is accepted or rejected.

Covered behaviours:

* ``OpenAICompatibleModel2._attachment_to_content_part`` is derived from
  the Chat Completions part plus ``_chat_parts_to_responses_parts``: the
  accepted MIME set and the wire shapes must match what v1's delegate
  sends for the same attachment.
* ``_consume_pending_call_id`` lost its unreachable trailing-run
  fallback; the remaining paths (pending queue, restored conversation,
  the error cases) are exercised through real tool turns.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.model import Attachment
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
)

_MODEL = "gpt-audit1012"

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Tool {name}",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    for name in ("f", "g")
]


def f() -> str:
    """Test-only tool stub."""
    return "F"


def g() -> str:
    """Test-only tool stub."""
    return "G"


_FUNCTION_MAP = {"f": f, "g": g}


def _function_call(call_id: str, name: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id or 'blank'}",
        "call_id": call_id,
        "name": name,
        "arguments": "{}",
    }


def _message(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


class _Script:
    """Answers each ``/v1/responses`` call with the next scripted turn.

    A turn is a list of ``(call_id, name)`` function calls; an empty list
    (or an exhausted script) answers with a plain assistant message.
    """

    def __init__(self, turns: list[list[tuple[str, str]]]) -> None:
        self.turns = list(turns)
        self.lock = threading.Lock()

    def __call__(self, request: Request) -> Reply:
        with self.lock:
            turn = self.turns.pop(0) if self.turns else []
            index = len(request.body.get("input", []))
        output = [_function_call(cid, name) for cid, name in turn] or [_message("ok")]
        return Reply(
            json_body={
                "id": f"resp_{index}",
                "object": "response",
                "created_at": 0,
                "model": _MODEL,
                "status": "completed",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "output": output,
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            }
        )


def _v2(
    base_url: str, prompt: str = "hi", attachments: list[Attachment] | None = None
) -> OpenAICompatibleModel2:
    model = OpenAICompatibleModel2(_MODEL, base_url=base_url, api_key="k")
    model.initialize(prompt, attachments=attachments)
    return model


def _tool_turn(model: OpenAICompatibleModel2) -> list[dict[str, Any]]:
    calls, _content, _response = model.generate_and_process_with_tools(
        _FUNCTION_MAP, tools_schema=_TOOLS
    )
    return calls


def _calls_and_outputs(items: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Reduce Responses input items to ``(kind, call_id, name_or_output)``."""
    reduced: list[tuple[str, str, str]] = []
    for item in items:
        if item.get("type") == "function_call":
            reduced.append(("call", item["call_id"], item["name"]))
        elif item.get("type") == "function_call_output":
            reduced.append(("output", item["call_id"], item["output"]))
    return reduced


def _sent_input(server: ScriptedOpenAIServer) -> list[dict[str, Any]]:
    """Return the ``input`` array of the last request the server received."""
    return list(server.requests[-1].body["input"])


# --------------------------------------------------------------------------
# Attachment conversion on the wire
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mime", "expected"),
    [
        ("image/png", "input_image"),
        ("image/jpeg", "input_image"),
        ("image/webp", "input_image"),
        ("image/gif", "input_image"),
        ("application/pdf", "input_file"),
        ("audio/mpeg", "input_audio"),
        ("audio/wav", "input_audio"),
    ],
)
def test_user_attachment_reaches_the_wire_in_responses_shape(mime: str, expected: str) -> None:
    att = Attachment(data=b"\x00\x01binary", mime_type=mime)
    with ScriptedOpenAIServer(_Script([])) as server:
        model = _v2(server.base_url, "look", [att])
        model.generate()
        user = _sent_input(server)[0]
    assert user["role"] == "user"
    part, text = user["content"]
    assert text == {"type": "input_text", "text": "look"}
    assert part["type"] == expected
    if expected == "input_image":
        assert part == {"type": "input_image", "image_url": att.to_data_url(), "detail": "auto"}
    elif expected == "input_file":
        assert part == {
            "type": "input_file",
            "filename": "attachment.pdf",
            "file_data": att.to_data_url(),
        }
    else:
        assert part["input_audio"]["data"] == att.to_base64()
        assert part["input_audio"]["format"] in ("mp3", "wav")


@pytest.mark.parametrize(
    "mime", ["image/png", "application/pdf", "audio/wav", "image/bmp", "text/plain"]
)
def test_v2_sends_the_same_user_parts_as_v1_delegating_to_responses(mime: str) -> None:
    """v2's part conversion must equal what v1's Responses delegate transmits."""
    att = Attachment(data=b"\x00\x01binary", mime_type=mime)
    with ScriptedOpenAIServer(_Script([])) as server:
        _v2(server.base_url, "look", [att]).generate()
        v2_user = _sent_input(server)[0]
        v1 = OpenAICompatibleModel(
            _MODEL,
            base_url=server.base_url,
            api_key="k",
            model_config={"reasoning_effort": "high", "use_responses_api": True},
        )
        v1.initialize("look", attachments=[att])
        v1.generate_and_process_with_tools(_FUNCTION_MAP, tools_schema=_TOOLS)
        assert server.requests[-1].path.endswith("/responses")
        v1_user = _sent_input(server)[0]
    assert v2_user == v1_user


@pytest.mark.parametrize("mime", ["image/bmp", "video/mp4", "audio/ogg", "text/plain"])
def test_rejected_attachments_never_reach_the_wire(mime: str) -> None:
    png = Attachment(data=b"png", mime_type="image/png")
    rejected = Attachment(data=b"x", mime_type=mime)
    with ScriptedOpenAIServer(_Script([])) as server:
        model = _v2(server.base_url, "look", [rejected, png])
        model.generate()
        sent = _sent_input(server)
    assert sent[0] == {
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": png.to_data_url(), "detail": "auto"},
            {"type": "input_text", "text": "look"},
        ],
    }
    assert sent[0] == model.conversation[0]


# --------------------------------------------------------------------------
# Pending-queue matching (same process, after a real tool turn)
# --------------------------------------------------------------------------


def test_pending_calls_are_answered_by_name_in_any_order() -> None:
    with ScriptedOpenAIServer(_Script([[("c1", "f"), ("c2", "g")]])) as server:
        model = _v2(server.base_url)
        assert [c["name"] for c in _tool_turn(model)] == ["f", "g"]
        model.add_function_results_to_conversation_and_return([("g", {"result": "G"})])
        assert model.conversation[-1] == {
            "type": "function_call_output",
            "call_id": "c2",
            "output": "G",
        }
        # Half-answered turn: the next generation is refused locally and
        # never reaches the server.
        with pytest.raises(KISSError, match="still pending"):
            model.generate()
        assert len(server.requests) == 1
        model.add_function_results_to_conversation_and_return([("f", {"result": "F"})])
        model.generate()
        assert _calls_and_outputs(_sent_input(server)) == [
            ("call", "c1", "f"),
            ("call", "c2", "g"),
            ("output", "c2", "G"),
            ("output", "c1", "F"),
        ]


def test_result_for_an_unknown_name_is_rejected_against_the_pending_queue() -> None:
    with ScriptedOpenAIServer(_Script([[("c1", "f")]])) as server:
        model = _v2(server.base_url)
        _tool_turn(model)
        before = list(model.conversation)
        with pytest.raises(KISSError, match="No pending function_call named 'h'"):
            model.add_function_results_to_conversation_and_return([("h", {"result": "H"})])
        assert model.conversation == before
        model.add_function_results_to_conversation_and_return([("f", {"result": "F"})])
        model.generate()
        assert _calls_and_outputs(_sent_input(server)) == [
            ("call", "c1", "f"),
            ("output", "c1", "F"),
        ]


def test_extra_result_beyond_the_pending_queue_is_rejected_and_rolled_back() -> None:
    with ScriptedOpenAIServer(_Script([[("c1", "f")]])) as server:
        model = _v2(server.base_url)
        _tool_turn(model)
        before = list(model.conversation)
        with pytest.raises(KISSError, match="No prior function_call found for tool result 'h'"):
            model.add_function_results_to_conversation_and_return(
                [("f", {"result": "F"}), ("h", {"result": "H"})]
            )
        # Atomic: the first result's output and its queue pop were undone,
        # so answering again produces exactly one output for c1.
        assert model.conversation == before
        model.add_function_results_to_conversation_and_return([("f", {"result": "F"})])
        model.generate()
        assert _calls_and_outputs(_sent_input(server)) == [
            ("call", "c1", "f"),
            ("output", "c1", "F"),
        ]


def test_explicit_tool_use_id_wins_but_still_consumes_the_pending_call() -> None:
    with ScriptedOpenAIServer(_Script([[("c1", "f"), ("c2", "f")]])) as server:
        model = _v2(server.base_url)
        _tool_turn(model)
        # FIFO matching would answer c1 first; the explicit ids reverse that.
        model.add_function_results_to_conversation_and_return(
            [
                ("f", {"result": "second", "tool_use_id": "c2"}),
                ("f", {"result": "first", "tool_use_id": "c1"}),
            ]
        )
        # Both pending entries were consumed, so the next turn is accepted.
        model.generate()
        assert _calls_and_outputs(_sent_input(server)) == [
            ("call", "c1", "f"),
            ("call", "c2", "f"),
            ("output", "c2", "second"),
            ("output", "c1", "first"),
        ]


def test_function_call_without_call_id_is_rejected_at_the_transport() -> None:
    """A pending entry with an empty ``call_id`` cannot exist.

    ``generate_and_process_with_tools`` refuses a ``function_call`` the
    server sent without a ``call_id`` before it is queued, which is why
    the ``empty call_id`` guard inside ``_consume_pending_call_id`` is not
    reachable through the public API and has no end-to-end test.
    """
    with ScriptedOpenAIServer(_Script([[("", "f")]])) as server:
        model = _v2(server.base_url)
        with pytest.raises(KISSError, match="function_call without call_id"):
            _tool_turn(model)
        assert len(server.requests) == 1


# --------------------------------------------------------------------------
# Restored conversation (pending queue lost, e.g. a model switch)
# --------------------------------------------------------------------------


def _restored(
    server: ScriptedOpenAIServer, source: OpenAICompatibleModel2
) -> OpenAICompatibleModel2:
    """Hand ``source``'s conversation to a fresh model.

    This is what ``KISSAgent`` does when it switches models mid-run
    (``kiss_agent.py``: ``new_model.conversation = old_conversation``): the
    new object has the items but none of the per-turn state.
    """
    model = OpenAICompatibleModel2(_MODEL, base_url=server.base_url, api_key="k")
    model.initialize("_")
    model.conversation = list(source.conversation)
    return model


def test_restored_conversation_matches_unanswered_calls_by_name() -> None:
    script = _Script([[("c1", "f")], [("c2", "g"), ("c3", "f")]])
    with ScriptedOpenAIServer(script) as server:
        source = _v2(server.base_url)
        _tool_turn(source)
        source.add_function_results_to_conversation_and_return([("f", {"result": "done"})])
        _tool_turn(source)
        model = _restored(server, source)
        model.add_function_results_to_conversation_and_return(
            [("f", {"result": "F"}), ("g", {"result": "G"})]
        )
        model.generate()
        # The answered c1 is skipped; f matches the unanswered c3, g matches c2.
        assert _calls_and_outputs(_sent_input(server)) == [
            ("call", "c1", "f"),
            ("output", "c1", "done"),
            ("call", "c2", "g"),
            ("call", "c3", "f"),
            ("output", "c3", "F"),
            ("output", "c2", "G"),
        ]


def test_restored_conversation_rejects_unknown_name() -> None:
    with ScriptedOpenAIServer(_Script([[("c2", "g")]])) as server:
        source = _v2(server.base_url)
        _tool_turn(source)
        model = _restored(server, source)
        before = list(model.conversation)
        with pytest.raises(KISSError, match="No unanswered function_call named 'h'"):
            model.add_function_results_to_conversation_and_return([("h", {"result": "H"})])
        assert model.conversation == before
        with pytest.raises(KISSError, match="still pending"):
            model.generate()
        assert len(server.requests) == 1


def test_result_without_any_prior_call_is_rejected() -> None:
    with ScriptedOpenAIServer(_Script([[("c1", "f")]])) as server:
        model = _v2(server.base_url)
        # No tool turn yet: nothing to match.
        with pytest.raises(KISSError, match="No prior function_call found for tool result 'f'"):
            model.add_function_results_to_conversation_and_return([("f", {"result": "F"})])
        assert model.conversation == [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ]
        # Fully answered history restored into a fresh model: still nothing.
        _tool_turn(model)
        model.add_function_results_to_conversation_and_return([("f", {"result": "F"})])
        restored = _restored(server, model)
        before = list(restored.conversation)
        with pytest.raises(KISSError, match="No prior function_call found for tool result 'f'"):
            restored.add_function_results_to_conversation_and_return([("f", {"result": "F"})])
        assert restored.conversation == before
        restored.generate()
        assert _calls_and_outputs(_sent_input(server)) == [
            ("call", "c1", "f"),
            ("output", "c1", "F"),
        ]
