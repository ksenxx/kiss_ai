# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests for OpenAI → Anthropic conversation hand-off.

When the Sorcar ``set_model`` tool switches a live agent from an OpenAI-schema
model (e.g. ``gpt-5.5``) to an Anthropic model (e.g. ``claude-*``), the raw
conversation is handed over: ``new_model.conversation = old_model.conversation``.
An OpenAI conversation stores assistant tool calls as ``tool_calls`` arrays,
tool results as ``role="tool"`` messages, and the system prompt as a
``role="system"`` message — none of which are valid Anthropic Messages-format
constructs.  Replaying those verbatim to the Anthropic Messages API fails with
``invalid_request_error`` (unexpected role / unknown field).

These tests verify that ``AnthropicModel`` converts such hand-off
conversations to Anthropic format before every API call:

* assistant ``tool_calls`` → ``tool_use`` content blocks,
* ``role="tool"`` messages → user messages with ``tool_result`` blocks
  (consecutive tool messages merged into one user turn),
* ``role="system"`` messages → hoisted into the top-level ``system`` param,
* ``image_url`` / ``file`` content parts → ``image`` / ``document`` blocks.
"""

import copy
from typing import Any

from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model_info import model
from kiss.tests.conftest import requires_anthropic_api_key

# An OpenAI-format conversation exactly as OpenAICompatibleModel stores it:
# a system message, an assistant message with a tool_calls array, and the
# tool result as a role="tool" message.
_OPENAI_STYLE_CONVERSATION: list[dict[str, Any]] = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Use the add tool to compute 2 + 3, then report the sum."},
    {
        "role": "assistant",
        "content": "Let me add those numbers.",
        "tool_calls": [
            {
                "id": "call_01",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_01", "content": "5"},
]


def _add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First addend.
        b: Second addend.

    Returns:
        The sum of a and b.
    """
    return a + b


def _make_offline_model() -> AnthropicModel:
    return AnthropicModel(
        model_name="claude-sonnet-4-20250514",
        api_key="test-key",
    )


class TestHandoffNormalization:
    """Offline behavior of the conversation normalizer on OpenAI input."""

    def test_tool_calls_become_tool_use_blocks(self) -> None:
        m = _make_offline_model()
        normalized = m._normalize_conversation_for_api(_OPENAI_STYLE_CONVERSATION)
        assistant = normalized[1]
        assert assistant["role"] == "assistant"
        assert "tool_calls" not in assistant
        assert assistant["content"] == [
            {"type": "text", "text": "Let me add those numbers."},
            {"type": "tool_use", "id": "call_01", "name": "add", "input": {"a": 2, "b": 3}},
        ]

    def test_tool_message_becomes_tool_result_user_message(self) -> None:
        m = _make_offline_model()
        normalized = m._normalize_conversation_for_api(_OPENAI_STYLE_CONVERSATION)
        tool_msg = normalized[2]
        assert tool_msg == {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_01", "content": "5"},
            ],
        }

    def test_system_message_is_removed_from_messages(self) -> None:
        m = _make_offline_model()
        normalized = m._normalize_conversation_for_api(_OPENAI_STYLE_CONVERSATION)
        assert all(msg["role"] != "system" for msg in normalized)
        assert normalized[0]["role"] == "user"

    def test_system_message_is_hoisted_into_system_param(self) -> None:
        m = _make_offline_model()
        m.conversation = copy.deepcopy(_OPENAI_STYLE_CONVERSATION)
        kwargs = m._build_create_kwargs()
        assert kwargs["system"] == "You are a concise assistant."

    def test_system_message_is_merged_with_system_instruction(self) -> None:
        m = AnthropicModel(
            model_name="claude-sonnet-4-20250514",
            api_key="test-key",
            model_config={"system_instruction": "Always be polite."},
        )
        m.conversation = copy.deepcopy(_OPENAI_STYLE_CONVERSATION)
        kwargs = m._build_create_kwargs()
        assert kwargs["system"] == "Always be polite.\n\nYou are a concise assistant."

    def test_duplicate_system_message_is_not_repeated(self) -> None:
        m = AnthropicModel(
            model_name="claude-sonnet-4-20250514",
            api_key="test-key",
            model_config={"system_instruction": "You are a concise assistant."},
        )
        m.conversation = copy.deepcopy(_OPENAI_STYLE_CONVERSATION)
        kwargs = m._build_create_kwargs()
        assert kwargs["system"] == "You are a concise assistant."

    def test_consecutive_tool_messages_merge_into_one_user_turn(self) -> None:
        m = _make_offline_model()
        conv: list[dict[str, Any]] = [
            {"role": "user", "content": "add twice"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 3, "b": 4}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "3"},
            {"role": "tool", "tool_call_id": "call_b", "content": "7"},
            {"role": "user", "content": "now summarize"},
        ]
        normalized = m._normalize_conversation_for_api(conv)
        assert len(normalized) == 3
        merged = normalized[2]
        assert merged["role"] == "user"
        assert merged["content"] == [
            {"type": "tool_result", "tool_use_id": "call_a", "content": "3"},
            {"type": "tool_result", "tool_use_id": "call_b", "content": "7"},
            {"type": "text", "text": "now summarize"},
        ]

    def test_empty_assistant_content_with_tool_calls(self) -> None:
        m = _make_offline_model()
        conv: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_x",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 1, "b": 1}'},
                    }
                ],
            },
        ]
        normalized = m._normalize_conversation_for_api(conv)
        assert normalized == [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_x", "name": "add", "input": {"a": 1, "b": 1}}
                ],
            }
        ]

    def test_invalid_tool_call_arguments_become_empty_input(self) -> None:
        m = _make_offline_model()
        conv: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {"name": "add", "arguments": "{not json"},
                    }
                ],
            },
        ]
        normalized = m._normalize_conversation_for_api(conv)
        assert normalized[0]["content"] == [
            {"type": "tool_use", "id": "call_bad", "name": "add", "input": {}}
        ]

    def test_empty_tool_message_content_is_omitted(self) -> None:
        m = _make_offline_model()
        conv: list[dict[str, Any]] = [
            {"role": "tool", "tool_call_id": "call_e", "content": ""},
        ]
        normalized = m._normalize_conversation_for_api(conv)
        assert normalized == [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_e"}],
            }
        ]

    def test_image_url_data_url_becomes_image_block(self) -> None:
        m = _make_offline_model()
        conv: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                    },
                    {"type": "text", "text": "describe"},
                ],
            }
        ]
        normalized = m._normalize_conversation_for_api(conv)
        assert normalized[0]["content"][0] == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="},
        }

    def test_image_url_http_url_becomes_url_source(self) -> None:
        m = _make_offline_model()
        conv: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.png"},
                    },
                    {"type": "text", "text": "describe"},
                ],
            }
        ]
        normalized = m._normalize_conversation_for_api(conv)
        assert normalized[0]["content"][0] == {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/cat.png"},
        }

    def test_file_part_becomes_document_block(self) -> None:
        m = _make_offline_model()
        conv: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {"file_data": "data:application/pdf;base64,cGRm"},
                    },
                    {"type": "text", "text": "summarize"},
                ],
            }
        ]
        normalized = m._normalize_conversation_for_api(conv)
        assert normalized[0]["content"][0] == {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "cGRm",
            },
        }

    def test_anthropic_style_messages_pass_through_unchanged(self) -> None:
        m = _make_offline_model()
        conv: list[dict[str, Any]] = [
            {"role": "user", "content": "Use the add tool to compute 2 + 3."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "call add", "signature": "sig"},
                    {"type": "text", "text": "Adding."},
                    {"type": "tool_use", "id": "toolu_01", "name": "add",
                     "input": {"a": 2, "b": 3}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01", "content": "5"},
                ],
            },
        ]
        assert m._normalize_conversation_for_api(copy.deepcopy(conv)) == conv


@requires_anthropic_api_key
class TestHandoffLive:
    """Live replay of an OpenAI-format history through the Anthropic API."""

    def test_generate_with_tools_after_openai_handoff(self) -> None:
        """The reciprocal failing scenario: tools call replaying an OpenAI history."""
        m = model("claude-haiku-4-5")
        assert isinstance(m, AnthropicModel)
        m.initialize("placeholder")
        m.conversation = copy.deepcopy(_OPENAI_STYLE_CONVERSATION)
        function_calls, content, _ = m.generate_and_process_with_tools({"add": _add})
        assert "5" in content or function_calls

    def test_generate_without_tools_after_openai_handoff(self) -> None:
        """generate() must also survive system/tool_calls/tool messages in history."""
        m = model("claude-haiku-4-5")
        assert isinstance(m, AnthropicModel)
        m.initialize("placeholder")
        m.conversation = copy.deepcopy(_OPENAI_STYLE_CONVERSATION) + [
            {"role": "user", "content": "Reply with exactly the word: done"},
        ]
        content, _ = m.generate()
        assert "done" in content.lower()
