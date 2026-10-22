# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end checks for the audit-G / audit-I fixes to the OpenAI transports.

* An embeddings endpoint that accepts the connection and never answers
  must raise within the per-request timeout, not park the task thread
  for the client's 1800 s.
* A tool turn delegated to the Responses API must store the model's own
  argument string in the chat ``tool_calls`` entry, not a re-encoding of
  the decoded dict.
* A reset of the chat model leaves its Responses delegate alone; the next
  delegated turn still goes through on the kept delegate.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.openai_compatible_model import EMBEDDINGS_TIMEOUT, OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    responses_event,
)

_MODEL = "audit-g-model-under-test"

# Written the way a model would: key order, trailing zero and spacing
# that ``json.dumps(json.loads(...))`` does not reproduce.
_RAW_ARGUMENTS = '{"text":"hi" , "count": 2.50}'

_FUNCTION_CALL_BODY = {
    "id": "resp-1",
    "object": "response",
    "created_at": 0,
    "model": _MODEL,
    "status": "completed",
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "output": [
        {
            "type": "function_call",
            "id": "fc-1",
            "call_id": "call-1",
            "name": "echo",
            "arguments": _RAW_ARGUMENTS,
            "status": "completed",
        }
    ],
    "usage": {
        "input_tokens": 5,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 2,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 7,
    },
}


def _echo(text: str, count: float = 1) -> str:
    """Tool under test."""
    return text * int(count)


def _function_call_responder(request: Request) -> Reply:
    """Answer every /responses request with one function call."""
    assert request.path.endswith("/responses"), request.path
    return Reply(json_body=_FUNCTION_CALL_BODY)


class TestEmbeddingsRequestIsBounded:
    """audit-I F2: a silent embeddings endpoint cannot hold the thread for 1800 s."""

    def test_silent_endpoint_raises_within_the_configured_timeout(self) -> None:
        """``get_embedding`` raises KISSError a few seconds after the request is queued."""
        gate = threading.Event()

        def never_answer(request: Request) -> Reply:
            assert request.path.endswith("/embeddings"), request.path
            return Reply(json_body={"data": []}, headers_gate=gate)

        with ScriptedOpenAIServer(never_answer) as server:
            try:
                model = OpenAICompatibleModel(
                    _MODEL,
                    base_url=server.base_url,
                    api_key="k",
                    model_config={"timeout": 0.5},
                )
                started = time.monotonic()
                with pytest.raises(KISSError, match="Embedding generation failed"):
                    model.get_embedding("hello")
                elapsed = time.monotonic() - started
            finally:
                gate.set()
        # 0.5 s, retried once by the SDK, plus its backoff: far below the
        # 60 s default and the client's 1800 s.
        assert elapsed < 10, f"get_embedding took {elapsed:.1f}s to give up"
        assert EMBEDDINGS_TIMEOUT < 1800

    def test_batch_request_is_bounded_too(self) -> None:
        """``get_embeddings`` shares the bounded request."""
        gate = threading.Event()

        def never_answer(request: Request) -> Reply:
            return Reply(json_body={"data": []}, headers_gate=gate)

        with ScriptedOpenAIServer(never_answer) as server:
            try:
                model = OpenAICompatibleModel(
                    _MODEL, base_url=server.base_url, api_key="k", model_config={"timeout": 0.5}
                )
                assert model.get_embeddings([]) == []
                started = time.monotonic()
                with pytest.raises(KISSError, match="Embedding generation failed"):
                    model.get_embeddings(["a", "b"])
                elapsed = time.monotonic() - started
            finally:
                gate.set()
        assert elapsed < 10, f"get_embeddings took {elapsed:.1f}s to give up"


def _embeddings_body(data: list[dict]) -> dict:
    """Return a syntactically valid embeddings response carrying *data*."""
    return {
        "object": "list",
        "model": _MODEL,
        "data": data,
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


class TestMalformedEmbeddingsResponseIsWrapped:
    """A response the SDK accepts but the conversion cannot use is a KISSError, not a TypeError."""

    def test_null_embedding_raises_kiss_error(self) -> None:
        """``"embedding": null`` is ``Embedding generation failed`` on both entry points."""

        def null_embedding(request: Request) -> Reply:
            assert request.path.endswith("/embeddings"), request.path
            return Reply(
                json_body=_embeddings_body([{"object": "embedding", "index": 0, "embedding": None}])
            )

        with ScriptedOpenAIServer(null_embedding) as server:
            model = OpenAICompatibleModel(_MODEL, base_url=server.base_url, api_key="k")
            with pytest.raises(KISSError, match=f"Embedding generation failed for model {_MODEL}"):
                model.get_embedding("hello")
            with pytest.raises(KISSError, match=f"Embedding generation failed for model {_MODEL}"):
                model.get_embeddings(["hello"])

    def test_batch_entry_without_index_raises_kiss_error(self) -> None:
        """A batch entry missing ``index`` cannot be ordered: wrapped the same way."""

        def no_index(request: Request) -> Reply:
            return Reply(
                json_body=_embeddings_body(
                    [
                        {"object": "embedding", "index": 1, "embedding": [0.1]},
                        {"object": "embedding", "embedding": [0.2]},
                    ]
                )
            )

        with ScriptedOpenAIServer(no_index) as server:
            model = OpenAICompatibleModel(_MODEL, base_url=server.base_url, api_key="k")
            with pytest.raises(KISSError, match="Embedding generation failed"):
                model.get_embeddings(["a", "b"])

    def test_well_formed_batch_is_returned_in_input_order(self) -> None:
        """Out-of-order entries are sorted by ``index``; the single-text path returns one vector."""

        def shuffled(request: Request) -> Reply:
            # One vector per input text, reversed so the sort has work to do.
            inputs = request.body["input"]
            count = 1 if isinstance(inputs, str) else len(inputs)
            entries = [
                {"object": "embedding", "index": i, "embedding": [i / 10, 0.0]}
                for i in reversed(range(count))
            ]
            return Reply(json_body=_embeddings_body(entries))

        def short(request: Request) -> Reply:
            return Reply(
                json_body=_embeddings_body(
                    [{"object": "embedding", "index": 0, "embedding": [0.0, 0.0]}]
                )
            )

        with ScriptedOpenAIServer(shuffled) as server:
            model = OpenAICompatibleModel(_MODEL, base_url=server.base_url, api_key="k")
            assert model.get_embeddings(["a", "b"]) == [[0.0, 0.0], [0.1, 0.0]]
            assert model.get_embedding("a") == [0.0, 0.0]
        with ScriptedOpenAIServer(short) as server:
            model = OpenAICompatibleModel(_MODEL, base_url=server.base_url, api_key="k")
            # A vector count that does not match the batch is malformed too.
            with pytest.raises(KISSError, match="Embedding generation failed"):
                model.get_embeddings(["a", "b"])


class TestStreamedTerminalEventWording:
    """The streamed terminal events keep their original error messages."""

    @pytest.mark.parametrize(
        ("event_type", "extra", "expected"),
        [
            (
                "response.incomplete",
                {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
                "Responses API stream ended incomplete: max_output_tokens",
            ),
            (
                "response.failed",
                {"status": "failed", "error": {"code": "x", "message": "boom"}},
                "Responses API stream ended with response.failed: boom",
            ),
        ],
    )
    def test_message_names_the_outcome(self, event_type: str, extra: dict, expected: str) -> None:
        """``response.incomplete`` reads "ended incomplete: <reason>", a failure "ended with"."""
        body = {**_FUNCTION_CALL_BODY, "output": [], **extra}

        def responder(request: Request) -> Reply:
            return Reply(sse_chunks=[responses_event(event_type, {"response": body})])

        with ScriptedOpenAIServer(responder) as server:
            model = OpenAICompatibleModel2(
                _MODEL, base_url=server.base_url, api_key="k", token_callback=print
            )
            model.initialize("hi")
            with pytest.raises(KISSError) as info:
                model.generate()
        assert str(info.value) == f"KISS Error: {expected}"


class TestDelegatedToolCallKeepsTheArgumentString:
    """audit-G F5: the chat ``tool_calls`` entry replays what the model wrote."""

    def test_tool_calls_entry_carries_the_verbatim_arguments(self) -> None:
        """The stored ``function.arguments`` is the raw string, decoded args reach the caller."""
        with ScriptedOpenAIServer(_function_call_responder) as server:
            model = OpenAICompatibleModel(
                _MODEL,
                base_url=server.base_url,
                api_key="k",
                model_config={"use_responses_api": True, "reasoning_effort": "high"},
            )
            model.initialize("call the tool")
            calls, content, _ = model.generate_and_process_with_tools({"echo": _echo})

        assert content == ""
        assert calls == [
            {"id": "call-1", "name": "echo", "arguments": {"text": "hi", "count": 2.5}}
        ]
        stored = model.conversation[-1]["tool_calls"]
        assert stored == [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "echo", "arguments": _RAW_ARGUMENTS},
            }
        ]
        assert stored[0]["function"]["arguments"] != json.dumps(calls[0]["arguments"])

    def test_next_turn_replays_the_verbatim_arguments(self) -> None:
        """The follow-up request sends the model's own string back, not a re-encoding."""
        with ScriptedOpenAIServer(_function_call_responder) as server:
            model = OpenAICompatibleModel(
                _MODEL,
                base_url=server.base_url,
                api_key="k",
                model_config={"use_responses_api": True, "reasoning_effort": "high"},
            )
            model.initialize("call the tool")
            model.generate_and_process_with_tools({"echo": _echo})
            model.add_function_results_to_conversation_and_return([("echo", {"result": "hihi"})])
            # Drop the cached raw items so the second request is rebuilt
            # from the chat conversation alone.
            model._delegate_raw_items.clear()
            model.generate_and_process_with_tools({"echo": _echo})
            sent = server.requests[-1].body["input"]

        replayed = [item for item in sent if item.get("type") == "function_call"]
        assert replayed and replayed[0]["arguments"] == _RAW_ARGUMENTS


class TestOwnerResetLeavesTheDelegateAlone:
    """audit-G F8: ``reset_conversation`` no longer resets the delegate."""

    def test_delegate_keeps_its_state_and_stays_usable(self) -> None:
        """The delegate's pending calls survive the owner's reset; the next turn works."""
        with ScriptedOpenAIServer(_function_call_responder) as server:
            model = OpenAICompatibleModel(
                _MODEL,
                base_url=server.base_url,
                api_key="k",
                model_config={"use_responses_api": True, "reasoning_effort": "high"},
            )
            model.initialize("turn one")
            model.generate_and_process_with_tools({"echo": _echo})
            delegate = model._responses_delegate
            assert delegate is not None
            pending = list(delegate._pending_function_calls)
            assert pending

            model.reset_conversation()
            assert model.conversation == []
            assert model._delegate_raw_items == {}
            assert delegate._pending_function_calls == pending

            model.initialize("turn two")
            calls, _content, _ = model.generate_and_process_with_tools({"echo": _echo})
            assert [c["name"] for c in calls] == ["echo"]
            assert model._responses_delegate is delegate
            # One connection pool: the delegate sends through the owner's client.
            assert delegate.client is model.client
            assert len(server.connection_keys) == 1
