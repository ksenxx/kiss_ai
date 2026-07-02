# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests for conversation hand-off to/from GeminiModel and CLI models.

When the Sorcar ``set_model`` tool switches a live agent between models, the raw
conversation is handed over: ``new_model.conversation = old_model.conversation``.
Each model class stores a different native format:

* ``OpenAICompatibleModel``: OpenAI Chat Completions messages — assistant
  ``tool_calls`` with JSON-**string** arguments, ``role="tool"`` results,
  ``role="system"`` messages.
* ``AnthropicModel``: Anthropic Messages content-block lists (``thinking`` /
  ``tool_use`` / ``tool_result`` blocks).
* ``GeminiModel``: OpenAI-like messages but with tool-call arguments stored as
  **dicts** and user messages optionally carrying an ``attachments`` key.
* ``ClaudeCodeModel`` / ``CodexModel``: flatten the conversation into one text
  prompt.

These tests verify every cross-format direction converts correctly:

* Gemini → OpenAI: dict arguments become JSON strings; ``attachments`` keys
  become content parts.
* Gemini → Anthropic: dict arguments become ``tool_use`` inputs; ``attachments``
  keys become ``image`` blocks.
* OpenAI → Gemini: JSON-string arguments are parsed to dict ``args``; ``system``
  messages are hoisted into ``system_instruction``; ``role="tool"`` results
  become ``function_response`` parts; ``image_url`` data URLs become inline
  bytes parts.
* Anthropic → Gemini: ``thinking`` blocks are dropped; ``text`` / ``tool_use`` /
  ``tool_result`` / ``image`` blocks become the matching Gemini parts.
* Any → CLI models: block lists are flattened to text (no dict reprs or
  base64 payloads in the prompt).
"""

import copy
import json
from typing import Any

from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.model import Attachment, flatten_content_to_text
from kiss.core.models.model_info import model
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.tests.conftest import (
    requires_anthropic_api_key,
    requires_gemini_api_key,
    requires_openai_api_key,
)

# A conversation exactly as GeminiModel stores it: OpenAI-like messages, but
# with tool-call arguments as a dict (not a JSON string).
_GEMINI_STYLE_CONVERSATION: list[dict[str, Any]] = [
    {"role": "user", "content": "Use the add tool to compute 2 + 3, then report the sum."},
    {
        "role": "assistant",
        "content": "Let me add those numbers.",
        "tool_calls": [
            {
                "id": "call_g1",
                "type": "function",
                "function": {"name": "add", "arguments": {"a": 2, "b": 3}},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_g1", "content": "5"},
]

# A conversation exactly as OpenAICompatibleModel stores it.
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

# A conversation exactly as AnthropicModel stores it.
_ANTHROPIC_STYLE_CONVERSATION: list[dict[str, Any]] = [
    {"role": "user", "content": "Use the add tool to compute 2 + 3, then report the sum."},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "I should call add.", "signature": "sig"},
            {"type": "text", "text": "Adding."},
            {"type": "tool_use", "id": "toolu_01", "name": "add", "input": {"a": 2, "b": 3}},
        ],
    },
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "5"}],
    },
]


def _parts(content: Any) -> list[Any]:
    """Return the non-None parts list of a Gemini ``Content``.

    Args:
        content: A ``google.genai.types.Content`` instance.

    Returns:
        The content's parts as a list.
    """
    assert content.parts is not None
    return list(content.parts)


def _function_calls(content: Any) -> list[Any]:
    """Return the non-None ``function_call`` payloads of a Gemini ``Content``.

    Args:
        content: A ``google.genai.types.Content`` instance.

    Returns:
        The ``FunctionCall`` objects of the content's parts.
    """
    return [p.function_call for p in _parts(content) if p.function_call is not None]


def _add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First addend.
        b: Second addend.

    Returns:
        The sum of a and b.
    """
    return a + b


def _make_gemini() -> GeminiModel:
    return GeminiModel(model_name="gemini-2.5-flash", api_key="test-key")


def _make_openai() -> OpenAICompatibleModel:
    return OpenAICompatibleModel(
        model_name="gpt-4o", base_url="https://api.openai.com/v1", api_key="test-key"
    )


def _make_anthropic() -> AnthropicModel:
    return AnthropicModel(model_name="claude-haiku-4-5", api_key="test-key")


class TestOpenAIToGemini:
    """OpenAI-format history converted to Gemini contents."""

    def test_json_string_arguments_become_dict_args(self) -> None:
        m = _make_gemini()
        m.conversation = copy.deepcopy(_OPENAI_STYLE_CONVERSATION)
        contents = m._convert_conversation_to_gemini_contents()
        assert len(contents) == 3  # system skipped
        assistant = contents[1]
        assert assistant.role == "model"
        fcs = _function_calls(assistant)
        assert len(fcs) == 1
        assert fcs[0].name == "add"
        assert fcs[0].args == {"a": 2, "b": 3}

    def test_tool_message_becomes_function_response_with_name(self) -> None:
        m = _make_gemini()
        m.conversation = copy.deepcopy(_OPENAI_STYLE_CONVERSATION)
        contents = m._convert_conversation_to_gemini_contents()
        tool_turn = contents[2]
        assert tool_turn.role == "user"
        fr = _parts(tool_turn)[0].function_response
        assert fr is not None
        assert fr.name == "add"
        assert fr.response == {"result": 5}

    def test_system_message_is_hoisted_into_system_instruction(self) -> None:
        m = _make_gemini()
        m.conversation = copy.deepcopy(_OPENAI_STYLE_CONVERSATION)
        config = m._build_config()
        assert config.system_instruction == "You are a concise assistant."

    def test_system_message_is_merged_with_configured_instruction(self) -> None:
        m = GeminiModel(
            model_name="gemini-2.5-flash",
            api_key="test-key",
            model_config={"system_instruction": "Always be polite."},
        )
        m.conversation = copy.deepcopy(_OPENAI_STYLE_CONVERSATION)
        config = m._build_config()
        assert config.system_instruction == (
            "Always be polite.\n\nYou are a concise assistant."
        )

    def test_duplicate_system_message_is_not_repeated(self) -> None:
        m = GeminiModel(
            model_name="gemini-2.5-flash",
            api_key="test-key",
            model_config={"system_instruction": "You are a concise assistant."},
        )
        m.conversation = copy.deepcopy(_OPENAI_STYLE_CONVERSATION)
        config = m._build_config()
        assert config.system_instruction == "You are a concise assistant."

    def test_image_url_data_url_becomes_inline_bytes(self) -> None:
        m = _make_gemini()
        m.conversation = [
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
        contents = m._convert_conversation_to_gemini_contents()
        parts = _parts(contents[0])
        assert parts[0].inline_data is not None
        assert parts[0].inline_data.data == b"hello"
        assert parts[0].inline_data.mime_type == "image/png"
        assert parts[1].text == "describe"


class TestAnthropicToGemini:
    """Anthropic-format history converted to Gemini contents."""

    def test_thinking_dropped_text_and_tool_use_converted(self) -> None:
        m = _make_gemini()
        m.conversation = copy.deepcopy(_ANTHROPIC_STYLE_CONVERSATION)
        contents = m._convert_conversation_to_gemini_contents()
        assistant = contents[1]
        assert assistant.role == "model"
        texts = [p.text for p in _parts(assistant) if p.text]
        assert texts == ["Adding."]  # thinking text is NOT replayed
        fcs = _function_calls(assistant)
        assert len(fcs) == 1
        assert fcs[0].name == "add"
        assert fcs[0].args == {"a": 2, "b": 3}

    def test_tool_result_block_becomes_function_response(self) -> None:
        m = _make_gemini()
        m.conversation = copy.deepcopy(_ANTHROPIC_STYLE_CONVERSATION)
        contents = m._convert_conversation_to_gemini_contents()
        tool_turn = contents[2]
        assert tool_turn.role == "user"
        fr = _parts(tool_turn)[0].function_response
        assert fr is not None
        assert fr.name == "add"  # looked up via the tool_use id map
        assert fr.response == {"result": 5}

    def test_nested_tool_result_content_text_is_extracted(self) -> None:
        m = _make_gemini()
        m.conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_x",
                        "content": [{"type": "text", "text": '{"sum": 5}'}],
                    }
                ],
            }
        ]
        contents = m._convert_conversation_to_gemini_contents()
        fr = _parts(contents[0])[0].function_response
        assert fr is not None
        assert fr.response == {"sum": 5}

    def test_anthropic_image_block_becomes_inline_bytes(self) -> None:
        m = _make_gemini()
        m.conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aGVsbG8=",
                        },
                    },
                    {"type": "text", "text": "describe"},
                ],
            }
        ]
        contents = m._convert_conversation_to_gemini_contents()
        parts = _parts(contents[0])
        assert parts[0].inline_data is not None
        assert parts[0].inline_data.data == b"hello"
        assert parts[0].inline_data.mime_type == "image/png"


class TestGeminiNativeFormat:
    """Gemini's own format must keep working unchanged."""

    def test_dict_arguments_pass_through(self) -> None:
        m = _make_gemini()
        m.conversation = copy.deepcopy(_GEMINI_STYLE_CONVERSATION)
        contents = m._convert_conversation_to_gemini_contents()
        assistant = contents[1]
        fcs = _function_calls(assistant)
        assert fcs[0].args == {"a": 2, "b": 3}
        fr = _parts(contents[2])[0].function_response
        assert fr is not None
        assert fr.name == "add"

    def test_user_attachments_become_inline_bytes(self) -> None:
        m = _make_gemini()
        m.conversation = [
            {
                "role": "user",
                "content": "look at this",
                "attachments": [Attachment(data=b"hello", mime_type="image/png")],
            }
        ]
        contents = m._convert_conversation_to_gemini_contents()
        parts = _parts(contents[0])
        assert parts[0].inline_data is not None
        assert parts[0].inline_data.data == b"hello"
        assert parts[1].text == "look at this"


class TestGeminiToOpenAI:
    """Gemini-format history converted to OpenAI Chat Completions messages."""

    def test_dict_arguments_become_json_strings(self) -> None:
        m = _make_openai()
        normalized = m._normalize_conversation_for_api(
            copy.deepcopy(_GEMINI_STYLE_CONVERSATION)
        )
        assistant = normalized[1]
        args = assistant["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str)
        assert json.loads(args) == {"a": 2, "b": 3}

    def test_attachments_key_is_lifted_into_content_parts(self) -> None:
        m = _make_openai()
        normalized = m._normalize_conversation_for_api(
            [
                {
                    "role": "user",
                    "content": "look at this",
                    "attachments": [Attachment(data=b"hello", mime_type="image/png")],
                }
            ]
        )
        msg = normalized[0]
        assert "attachments" not in msg
        assert msg["content"][0] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aGVsbG8="},
        }
        assert msg["content"][1] == {"type": "text", "text": "look at this"}

    def test_openai_native_string_arguments_unchanged(self) -> None:
        m = _make_openai()
        normalized = m._normalize_conversation_for_api(
            copy.deepcopy(_OPENAI_STYLE_CONVERSATION)
        )
        assistant = normalized[2]
        assert assistant["tool_calls"][0]["function"]["arguments"] == '{"a": 2, "b": 3}'


class TestGeminiToAnthropic:
    """Gemini-format history converted to Anthropic Messages format."""

    def test_dict_arguments_become_tool_use_input(self) -> None:
        m = _make_anthropic()
        normalized = m._normalize_conversation_for_api(
            copy.deepcopy(_GEMINI_STYLE_CONVERSATION)
        )
        assistant = normalized[1]
        assert assistant["content"] == [
            {"type": "text", "text": "Let me add those numbers."},
            {"type": "tool_use", "id": "call_g1", "name": "add", "input": {"a": 2, "b": 3}},
        ]
        tool_turn = normalized[2]
        assert tool_turn == {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_g1", "content": "5"},
            ],
        }

    def test_attachments_key_is_lifted_into_image_block(self) -> None:
        m = _make_anthropic()
        normalized = m._normalize_conversation_for_api(
            [
                {
                    "role": "user",
                    "content": "look at this",
                    "attachments": [Attachment(data=b"hello", mime_type="image/png")],
                }
            ]
        )
        msg = normalized[0]
        assert "attachments" not in msg
        assert msg["content"][0] == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="},
        }
        assert msg["content"][1] == {"type": "text", "text": "look at this"}


class TestCliPromptFlattening:
    """Foreign block-list content flattened to text for the CLI models."""

    def test_flatten_anthropic_blocks(self) -> None:
        text = flatten_content_to_text(_ANTHROPIC_STYLE_CONVERSATION[1]["content"])
        assert "Adding." in text
        assert '[Tool Call] add({"a": 2, "b": 3})' in text
        assert "thinking" not in text  # hidden provider state dropped

    def test_flatten_tool_result_and_media_blocks(self) -> None:
        text = flatten_content_to_text(
            [
                {"type": "tool_result", "tool_use_id": "t1", "content": "5"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="},
                },
            ]
        )
        assert "5" in text
        assert "aGVsbG8=" not in text  # no base64 dumps in the prompt
        assert "[image attachment omitted]" in text

    def test_claude_code_prompt_has_no_block_reprs(self) -> None:
        m = ClaudeCodeModel(model_name="cc/haiku")
        m.conversation = copy.deepcopy(_ANTHROPIC_STYLE_CONVERSATION)
        prompt = m._build_prompt()
        assert "'type':" not in prompt
        assert "Adding." in prompt
        # The Anthropic tool_result lives in a user turn, so its text is
        # rendered under the [User] label.
        assert "[User]: 5" in prompt


@requires_openai_api_key
class TestGeminiToOpenAILive:
    """Live replay of a Gemini-format history through the OpenAI API."""

    def test_generate_with_tools_after_gemini_handoff(self) -> None:
        m = model("gpt-4o")
        assert isinstance(m, OpenAICompatibleModel)
        m.initialize("placeholder")
        m.conversation = copy.deepcopy(_GEMINI_STYLE_CONVERSATION)
        function_calls, content, _ = m.generate_and_process_with_tools({"add": _add})
        assert "5" in content or function_calls

    def test_generate_without_tools_after_gemini_handoff(self) -> None:
        m = model("gpt-4o")
        assert isinstance(m, OpenAICompatibleModel)
        m.initialize("placeholder")
        m.conversation = copy.deepcopy(_GEMINI_STYLE_CONVERSATION) + [
            {"role": "user", "content": "Reply with exactly the word: done"},
        ]
        content, _ = m.generate()
        assert "done" in content.lower()


@requires_anthropic_api_key
class TestGeminiToAnthropicLive:
    """Live replay of a Gemini-format history through the Anthropic API."""

    def test_generate_with_tools_after_gemini_handoff(self) -> None:
        m = model("claude-haiku-4-5")
        assert isinstance(m, AnthropicModel)
        m.initialize("placeholder")
        m.conversation = copy.deepcopy(_GEMINI_STYLE_CONVERSATION)
        function_calls, content, _ = m.generate_and_process_with_tools({"add": _add})
        assert "5" in content or function_calls


@requires_gemini_api_key
class TestToGeminiLive:
    """Live replay of OpenAI- and Anthropic-format histories through Gemini."""

    def test_generate_with_tools_after_openai_handoff(self) -> None:
        m = model("gemini-2.5-flash")
        assert isinstance(m, GeminiModel)
        m.initialize("placeholder")
        m.conversation = copy.deepcopy(_OPENAI_STYLE_CONVERSATION)
        function_calls, content, _ = m.generate_and_process_with_tools({"add": _add})
        assert "5" in content or function_calls

    def test_generate_without_tools_after_anthropic_handoff(self) -> None:
        m = model("gemini-2.5-flash")
        assert isinstance(m, GeminiModel)
        m.initialize("placeholder")
        m.conversation = copy.deepcopy(_ANTHROPIC_STYLE_CONVERSATION) + [
            {"role": "user", "content": "Reply with exactly the word: done"},
        ]
        content, _ = m.generate()
        assert "done" in content.lower()
