# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the OpenAI-compatible transports (sweep 2026-10-26).

Every test talks to a real ``ThreadingHTTPServer`` through the real OpenAI
SDK and asserts on what reached the wire or on the conversation shape:

* ``extra_headers`` from ``model_config`` is installed once, as the client's
  default headers, and still reaches every endpoint; a header replaced
  between turns reaches the next request on every path (the client is
  re-synced with the config before each request and rebuilt only then).
* The DeepSeek text-based tool turn is recorded through the same
  ``_append_assistant_turn`` as a native tool turn (v1) and as
  ``function_call`` items (v2), and the follow-up request replays it.
* A delegated Responses turn with several calls caches its raw items once,
  under the first call id, and the follow-up request replays the cache.
* A hand-off assistant message whose ``tool_calls`` carry dict arguments is
  replayed as a ``function_call`` item with JSON-string arguments.
* A first user message with ``None`` content: v1 normalises (and drops it)
  before merging the tools prompt; v2 merges first, and the merged text
  must start with the prompt, not with ``"None"``.
* The v2 ``reasoning`` request block: a configured ``reasoning_effort``
  lands as ``effort`` with ``summary: auto`` unless a summary is configured;
  a configured ``reasoning`` dict with an effort gets the default summary; a
  ``reasoning`` dict without an effort goes out untouched.
* An Anthropic hand-off ``tool_use`` block and an attachment-bearing
  message normalise to the Chat Completions shapes the shared helpers build.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from kiss.core.models.model import Attachment
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401
from kiss.tests.core.models.openai_sse_harness import Reply, Request, ScriptedOpenAIServer

_DEEPSEEK = "openrouter/deepseek/deepseek-r1"
_TOOL_CALL_TEXT = "<think>which tool</think>Running it.\n" + json.dumps(
    {"tool_calls": [{"name": "echo", "arguments": {"text": "hi"}}]}
)
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo text.",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
        },
    }
]
_USAGE_CHAT = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
_USAGE_RESPONSES = {
    "input_tokens": 5,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 4,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 9,
}


def echo(text: str) -> str:
    """Return *text* (test-only tool)."""
    return text


def _chat_reply(text: str) -> Reply:
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek-r1",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": _USAGE_CHAT,
    }
    return Reply(json_body=body)


def _responses_reply(output: list[dict[str, Any]]) -> Reply:
    body = {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-under-test",
        "status": "completed",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": output,
        "usage": _USAGE_RESPONSES,
    }
    return Reply(json_body=body)


def _message_output(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _function_call_output(call_id: str, text: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": "echo",
        "arguments": json.dumps({"text": text}),
    }


def _two_call_turn(request: Request) -> Reply:
    """Answer a Responses request with a two-call tool turn, or a plain text turn."""
    if any(item.get("type") == "function_call_output" for item in request.body.get("input", [])):
        return _responses_reply([_message_output("done")])
    return _responses_reply(
        [_function_call_output("call_a", "one"), _function_call_output("call_b", "two")]
    )


def _delegating_model(base_url: str) -> OpenAICompatibleModel:
    model = OpenAICompatibleModel(
        "gpt-under-test",
        base_url=base_url,
        api_key="test-key",
        model_config={
            "use_responses_api": True,
            "reasoning_effort": "low",
            "extra_headers": {"X-Sweep": "on"},
        },
    )
    model.initialize("go")
    return model


class TestExtraHeadersInstalledOnce:
    """``extra_headers`` is a client default, not also a per-request kwarg."""

    def test_v1_chat_sends_the_header_and_forwards_no_request_param(self) -> None:
        with ScriptedOpenAIServer(lambda req: _chat_reply("ok")) as server:
            model = OpenAICompatibleModel(
                _DEEPSEEK,
                base_url=server.base_url,
                api_key="test-key",
                model_config={"extra_headers": {"X-Sweep": "on"}, "temperature": 0.2},
            )
            model.initialize("hi")
            assert model.generate()[0] == "ok"
            (request,) = server.requests
        assert request.headers["x-sweep"] == "on"
        assert request.body["temperature"] == 0.2
        assert model._config_request_params() == {"temperature": 0.2}

    def test_v1_delegated_turn_and_v2_send_the_header(self) -> None:
        with ScriptedOpenAIServer(_two_call_turn) as server:
            model = _delegating_model(server.base_url)
            model.generate_and_process_with_tools({"echo": echo}, tools_schema=_TOOLS)
            model2 = OpenAICompatibleModel2(
                "gpt-under-test",
                base_url=server.base_url,
                api_key="test-key",
                model_config={"extra_headers": {"X-Sweep": "two"}, "reasoning_effort": "low"},
            )
            model2.initialize("hi")
            model2.generate_and_process_with_tools({"echo": echo}, tools_schema=_TOOLS)
            first, second = server.requests
        assert first.path.endswith("/responses") and first.headers["x-sweep"] == "on"
        assert second.path.endswith("/responses") and second.headers["x-sweep"] == "two"
        assert "extra_headers" not in model2._config_request_params()


def _plain_responses_turn(request: Request) -> Reply:
    return _responses_reply([_message_output("ok")])


class TestExtraHeadersReplacedBetweenTurns:
    """A header replaced in ``model_config`` between turns reaches the next request.

    Every request path re-syncs the client with the current config before
    sending (and keeps the client when nothing changed), so the second turn
    carries the replacement on Chat Completions, on direct Responses and on
    the delegated Responses path alike.
    """

    @staticmethod
    def _three_turns(server: ScriptedOpenAIServer, model: Any, generate: Any, path: str) -> None:
        model.model_config["extra_headers"] = {"X-Sweep": "one"}
        model.initialize("hi")
        generate()
        first_client = model.client
        generate()
        unchanged_client = model.client
        model.model_config["extra_headers"] = {"X-Sweep": "two"}
        generate()
        assert [r.headers["x-sweep"] for r in server.requests] == ["one", "one", "two"]
        assert all(r.path.endswith(path) for r in server.requests)
        assert unchanged_client is first_client, "unchanged inputs must keep the client"
        assert model.client is not first_client, "a changed header must rebuild the client"

    def test_v1_chat_completions(self) -> None:
        with ScriptedOpenAIServer(lambda req: _chat_reply("ok")) as server:
            model = OpenAICompatibleModel(_DEEPSEEK, base_url=server.base_url, api_key="k")
            self._three_turns(server, model, model.generate, "/chat/completions")

    def test_v2_direct_responses(self) -> None:
        with ScriptedOpenAIServer(_plain_responses_turn) as server:
            model = OpenAICompatibleModel2("gpt-under-test", base_url=server.base_url, api_key="k")
            self._three_turns(server, model, model.generate, "/responses")

    def test_v1_delegated_responses(self) -> None:
        with ScriptedOpenAIServer(_plain_responses_turn) as server:
            model = OpenAICompatibleModel(
                "gpt-under-test",
                base_url=server.base_url,
                api_key="k",
                model_config={"use_responses_api": True, "reasoning_effort": "low"},
            )

            def tool_turn() -> None:
                model.generate_and_process_with_tools({"echo": echo}, _TOOLS)

            self._three_turns(server, model, tool_turn, "/responses")
            delegate = model._responses_delegate
            assert delegate is not None and delegate.client is model.client


class TestTextBasedToolTurnShape:
    """The DeepSeek text-tools turn is recorded like a native one and replays."""

    def test_v1_records_tool_calls_and_replays_them(self) -> None:
        replies = iter([_chat_reply(_TOOL_CALL_TEXT), _chat_reply("done")])
        with ScriptedOpenAIServer(lambda req: next(replies)) as server:
            model = OpenAICompatibleModel(_DEEPSEEK, base_url=server.base_url, api_key="k")
            model.initialize("hi")
            calls, content, _ = model.generate_and_process_with_tools({"echo": echo})
            assert [c["name"] for c in calls] == ["echo"]
            assert content.startswith("<think>")
            assistant = model.conversation[-1]
            assert assistant["role"] == "assistant"
            assert assistant["content"] == "Running it.\n" + json.dumps(
                {"tool_calls": [{"name": "echo", "arguments": {"text": "hi"}}]}
            )
            (tool_call,) = assistant["tool_calls"]
            assert tool_call["id"] == calls[0]["id"] and tool_call["type"] == "function"
            assert json.loads(tool_call["function"]["arguments"]) == {"text": "hi"}
            model.add_function_results_to_conversation_and_return([("echo", {"result": "hi"})])
            assert model.generate()[0] == "done"
            _, second = server.requests
        roles = [m["role"] for m in second.body["messages"]]
        assert roles == ["user", "assistant", "tool"]
        assert second.body["messages"][1]["tool_calls"] == assistant["tool_calls"]
        assert second.body["messages"][2]["tool_call_id"] == tool_call["id"]

    def test_v1_plain_text_turn_records_no_tool_calls_key(self) -> None:
        with ScriptedOpenAIServer(lambda req: _chat_reply("just text")) as server:
            model = OpenAICompatibleModel(_DEEPSEEK, base_url=server.base_url, api_key="k")
            model.initialize("hi")
            calls, _content, _ = model.generate_and_process_with_tools({"echo": echo})
        assert calls == []
        assert model.conversation[-1] == {"role": "assistant", "content": "just text"}

    def test_v2_records_function_call_items_and_replays_them(self) -> None:
        replies = iter(
            [
                _responses_reply([_message_output(_TOOL_CALL_TEXT)]),
                _responses_reply([_message_output("done")]),
            ]
        )
        with ScriptedOpenAIServer(lambda req: next(replies)) as server:
            model = OpenAICompatibleModel2(_DEEPSEEK, base_url=server.base_url, api_key="k")
            model.initialize("hi")
            calls, _content, _ = model.generate_and_process_with_tools({"echo": echo})
            assert [c["name"] for c in calls] == ["echo"]
            text_item, call_item = model.conversation[-2:]
            assert text_item["role"] == "assistant" and "<think>" not in text_item["content"]
            assert call_item == {
                "type": "function_call",
                "call_id": calls[0]["id"],
                "name": "echo",
                "arguments": json.dumps({"text": "hi"}),
            }
            model.add_function_results_to_conversation_and_return([("echo", {"result": "hi"})])
            assert model.generate()[0] == "done"
            _, second = server.requests
        items = second.body["input"]
        assert [i.get("type") or i["role"] for i in items[-3:]] == [
            "assistant",
            "function_call",
            "function_call_output",
        ]
        assert items[-2] == call_item
        assert items[-1]["call_id"] == calls[0]["id"]


class TestDelegatedTurnCache:
    """A multi-call delegated turn caches its raw items once and replays them."""

    def test_raw_items_keyed_by_first_call_id_and_replayed(self) -> None:
        with ScriptedOpenAIServer(_two_call_turn) as server:
            model = _delegating_model(server.base_url)
            calls, _content, _ = model.generate_and_process_with_tools(
                {"echo": echo}, tools_schema=_TOOLS
            )
            assert [c["id"] for c in calls] == ["call_a", "call_b"]
            assert list(model._delegate_raw_items) == ["call_a"]
            model.add_function_results_to_conversation_and_return(
                [("echo", {"result": "one"}), ("echo", {"result": "two"})]
            )
            model.generate_and_process_with_tools({"echo": echo}, tools_schema=_TOOLS)
            assert list(model._delegate_raw_items) == ["call_a"]
            _, second = server.requests
        replayed = [i for i in second.body["input"] if i.get("type") == "function_call"]
        assert [(i["call_id"], i["id"]) for i in replayed] == [
            ("call_a", "fc_call_a"),
            ("call_b", "fc_call_b"),
        ]
        outputs = [i for i in second.body["input"] if i.get("type") == "function_call_output"]
        assert [o["call_id"] for o in outputs] == ["call_a", "call_b"]

    def test_handed_off_dict_arguments_are_replayed_as_json(self) -> None:
        with ScriptedOpenAIServer(_two_call_turn) as server:
            model = _delegating_model(server.base_url)
            model.conversation.extend(
                [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_x",
                                "type": "function",
                                "function": {"name": "echo", "arguments": {"text": "x"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_x", "content": "x"},
                ]
            )
            model.generate_and_process_with_tools({"echo": echo}, tools_schema=_TOOLS)
            (request,) = server.requests
        (replayed,) = [i for i in request.body["input"] if i.get("type") == "function_call"]
        assert replayed == {
            "type": "function_call",
            "call_id": "call_x",
            "name": "echo",
            "arguments": json.dumps({"text": "x"}),
        }


class TestResponsesReasoningBlock:
    """One pass builds the ``reasoning`` block the Responses request carries."""

    def _request_with(self, model_config: dict[str, Any]) -> Request:
        with ScriptedOpenAIServer(lambda req: _responses_reply([_message_output("ok")])) as server:
            model = OpenAICompatibleModel2(
                "gpt-under-test", base_url=server.base_url, api_key="k", model_config=model_config
            )
            model.initialize("hi")
            assert model.generate()[0] == "ok"
            (request,) = server.requests
        return request

    def test_effort_gets_the_default_summary(self) -> None:
        request = self._request_with({"reasoning_effort": "high"})
        assert request.body["reasoning"] == {"effort": "high", "summary": "auto"}

    def test_effort_overrides_a_configured_dict_and_keeps_its_summary(self) -> None:
        request = self._request_with(
            {"reasoning_effort": "high", "reasoning": {"effort": "low", "summary": "detailed"}}
        )
        assert request.body["reasoning"] == {"effort": "high", "summary": "detailed"}

    def test_configured_dict_with_effort_gets_the_default_summary(self) -> None:
        request = self._request_with({"reasoning": {"effort": "medium"}})
        assert request.body["reasoning"] == {"effort": "medium", "summary": "auto"}

    def test_configured_dict_without_effort_is_untouched(self) -> None:
        request = self._request_with({"reasoning": {"summary": "concise"}})
        assert request.body["reasoning"] == {"summary": "concise"}

    def test_no_reasoning_config_sends_no_block(self) -> None:
        request = self._request_with({})
        assert "reasoning" not in request.body


class TestAnthropicHandoffNormalisation:
    """Hand-off ``tool_use`` blocks and attachments normalise through the shared helpers."""

    def test_tool_use_block_becomes_a_chat_tool_call_entry(self) -> None:
        with ScriptedOpenAIServer(lambda req: _chat_reply("ok")) as server:
            model = OpenAICompatibleModel(_DEEPSEEK, base_url=server.base_url, api_key="k")
            model.initialize("hi")
            model.conversation.extend(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "calling"},
                            {
                                "type": "tool_use",
                                "id": "tu_1",
                                "name": "echo",
                                "input": {"text": "x"},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "x"}],
                    },
                ]
            )
            assert model.generate()[0] == "ok"
            (request,) = server.requests
        _user, assistant, tool = request.body["messages"]
        assert assistant["content"] == "calling"
        assert assistant["tool_calls"] == [
            {
                "id": "tu_1",
                "type": "function",
                "function": {"name": "echo", "arguments": json.dumps({"text": "x"})},
            }
        ]
        assert tool == {"role": "tool", "tool_call_id": "tu_1", "content": "x"}

    def test_attachment_message_keeps_its_text_as_a_text_part(self) -> None:
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
        with ScriptedOpenAIServer(lambda req: _chat_reply("ok")) as server:
            model = OpenAICompatibleModel(_DEEPSEEK, base_url=server.base_url, api_key="k")
            model.initialize("hi")
            model.conversation.append(
                {"role": "user", "content": "look", "attachments": [Attachment(png, "image/png")]}
            )
            assert model.generate()[0] == "ok"
            (request,) = server.requests
        parts = request.body["messages"][1]["content"]
        assert parts[0]["type"] == "image_url"
        assert parts[-1] == {"type": "text", "text": "look"}


class TestToolsPromptOnTextlessFirstMessage:
    """A ``None``-content first message never puts ``"None"`` on the wire."""

    def test_v1_drops_the_message_and_leads_with_the_prompt(self) -> None:
        with ScriptedOpenAIServer(lambda req: _chat_reply("ok")) as server:
            model = OpenAICompatibleModel(_DEEPSEEK, base_url=server.base_url, api_key="k")
            model.initialize("hi")
            model.conversation[0]["content"] = None
            model.add_message_to_conversation("user", "second")
            model.generate_and_process_with_tools({"echo": echo})
            (request,) = server.requests
        messages = request.body["messages"]
        assert [m["role"] for m in messages] == ["user"]
        assert messages[0]["content"].startswith("second\n")
        assert "None" not in messages[0]["content"]

    def test_v2_merges_the_prompt_without_a_none_prefix(self) -> None:
        with ScriptedOpenAIServer(lambda req: _responses_reply([_message_output("ok")])) as server:
            model = OpenAICompatibleModel2(_DEEPSEEK, base_url=server.base_url, api_key="k")
            model.initialize("hi")
            model.conversation[0]["content"] = None
            model.add_message_to_conversation("user", "second")
            model.generate_and_process_with_tools({"echo": echo})
            (request,) = server.requests
        first, second = request.body["input"]
        assert first["role"] == "user" and second["content"] == "second"
        assert first["content"].startswith("\n\n## Available Tools")
        assert "None" not in first["content"]
