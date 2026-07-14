# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end regression tests for provider model simplifications.

Pins the offline-testable behavior of AnthropicModel, GeminiModel,
CodexModel, and ClaudeCodeModel internals (message conversion, payload
construction, token counting, config handling) before and after
behavior-preserving simplifications.  No mocks/patches — only real
function calls and real object construction; no network calls.
"""

import base64
import json
import os
import stat

import pytest
from google.genai import types as gtypes

from kiss.core.kiss_error import KISSError
from kiss.core.models.anthropic_model import (
    AnthropicModel,
    _openai_part_to_anthropic_block,
    _parse_data_url,
    _tool_calls_to_tool_use_blocks,
    _uses_adaptive_thinking,
)
from kiss.core.models.anthropic_model import (
    _attachments_to_blocks as anthropic_attachments_to_blocks,
)
from kiss.core.models.claude_code_model import (
    ClaudeCodeModel,
    _claude_code_cache_creation_tokens,
    _find_consecutive_tool_calls_end,
)
from kiss.core.models.codex_model import (
    CodexModel,
    _find_in_candidate_paths,
)
from kiss.core.models.gemini_model import (
    GeminiModel,
    _coerce_args_dict,
    _decode_base64,
    _media_block_to_part,
    _tool_result_response_dict,
)
from kiss.core.models.model import Attachment, encode_binary_attachment

PNG_BYTES = b"\x89PNG\r\n\x1a\nfakepngdata"
PDF_BYTES = b"%PDF-1.4 fake pdf"
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")
PDF_B64 = base64.b64encode(PDF_BYTES).decode("ascii")


def make_anthropic() -> AnthropicModel:
    """Build an AnthropicModel with a dummy key (no network)."""
    return AnthropicModel("claude-sonnet-4-5", api_key="test")


class TestAnthropicHelpers:
    """Module-level conversion helpers in anthropic_model."""

    def test_parse_data_url_valid(self) -> None:
        assert _parse_data_url(f"data:image/png;base64,{PNG_B64}") == ("image/png", PNG_B64)

    def test_parse_data_url_no_media_type(self) -> None:
        assert _parse_data_url(f"data:;base64,{PNG_B64}") == (
            "application/octet-stream",
            PNG_B64,
        )

    def test_parse_data_url_rejects_non_data(self) -> None:
        assert _parse_data_url("https://example.com/x.png") is None

    def test_parse_data_url_rejects_non_base64(self) -> None:
        assert _parse_data_url("data:text/plain,hello") is None

    def test_openai_image_url_data_part(self) -> None:
        block = _openai_part_to_anthropic_block(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}}
        )
        assert block == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64},
        }

    def test_openai_image_url_remote_part(self) -> None:
        block = _openai_part_to_anthropic_block(
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
        )
        assert block == {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/a.png"},
        }

    def test_openai_image_url_empty_dropped(self) -> None:
        assert _openai_part_to_anthropic_block({"type": "image_url", "image_url": {}}) is None

    def test_openai_file_part(self) -> None:
        block = _openai_part_to_anthropic_block(
            {"type": "file", "file": {"file_data": f"data:application/pdf;base64,{PDF_B64}"}}
        )
        assert block == {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": PDF_B64},
        }

    def test_openai_file_part_bad_data_dropped(self) -> None:
        assert _openai_part_to_anthropic_block({"type": "file", "file": {}}) is None

    def test_unknown_part_dropped(self) -> None:
        assert _openai_part_to_anthropic_block({"type": "bogus"}) is None

    def test_tool_calls_to_tool_use_blocks_dicts(self) -> None:
        blocks = _tool_calls_to_tool_use_blocks(
            [
                {
                    "id": "call_1",
                    "function": {"name": "Bash", "arguments": '{"command": "ls"}'},
                },
                {"id": "call_2", "function": {"name": "Read", "arguments": "not json"}},
                {"id": "call_3", "function": {"name": "F", "arguments": '["x"]'}},
                {"id": "call_4", "function": {"name": "G", "arguments": {"a": 1}}},
                {"id": "call_5", "function": {"name": "H", "arguments": "   "}},
            ]
        )
        assert blocks == [
            {"type": "tool_use", "id": "call_1", "name": "Bash", "input": {"command": "ls"}},
            {"type": "tool_use", "id": "call_2", "name": "Read", "input": {}},
            {"type": "tool_use", "id": "call_3", "name": "F", "input": {}},
            {"type": "tool_use", "id": "call_4", "name": "G", "input": {"a": 1}},
            {"type": "tool_use", "id": "call_5", "name": "H", "input": {}},
        ]

    def test_attachments_to_blocks_image_pdf_video(self) -> None:
        blocks = anthropic_attachments_to_blocks(
            [
                Attachment(data=PNG_BYTES, mime_type="image/png"),
                Attachment(data=PDF_BYTES, mime_type="application/pdf"),
                Attachment(data=b"vid", mime_type="video/mp4"),
            ]
        )
        assert blocks == [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64},
            },
            {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": PDF_B64},
            },
        ]

    def test_uses_adaptive_thinking_boundaries(self) -> None:
        assert _uses_adaptive_thinking("claude-opus-4-6") is True
        assert _uses_adaptive_thinking("claude-opus-4-7-20260101") is True
        assert _uses_adaptive_thinking("claude-opus-4-5") is False
        assert _uses_adaptive_thinking("claude-opus-4-1") is False
        assert _uses_adaptive_thinking("claude-sonnet-4-6") is False
        assert _uses_adaptive_thinking("claude-opus-4-x") is False


class TestAnthropicNormalization:
    """Conversation normalization and payload construction."""

    def test_normalize_content_blocks_drops_whitespace_text(self) -> None:
        m = make_anthropic()
        blocks = m._normalize_content_blocks(
            [
                {"type": "text", "text": "   "},
                {"type": "text", "text": "keep"},
                {"type": "tool_use", "id": "t1", "name": "F", "input": {"a": 1}},
            ]
        )
        assert blocks == [
            {"type": "text", "text": "keep"},
            {"type": "tool_use", "id": "t1", "name": "F", "input": {"a": 1}},
        ]

    def test_normalize_content_blocks_converts_openai_parts(self) -> None:
        m = make_anthropic()
        blocks = m._normalize_content_blocks(
            [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}},
                {"type": "input_audio", "input_audio": {}},
            ]
        )
        assert len(blocks) == 1
        assert blocks[0]["type"] == "image"

    def test_extract_text_from_blocks(self) -> None:
        m = make_anthropic()
        text = m._extract_text_from_blocks(
            [
                {"type": "text", "text": "a"},
                {"type": "tool_use", "id": "1", "name": "F", "input": {}},
                {"type": "text", "text": "b"},
            ]
        )
        assert text == "ab"

    def test_normalize_conversation_system_dropped_tool_converted(self) -> None:
        m = make_anthropic()
        conv: list[dict[str, object]] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "calling",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "F", "arguments": '{"x": 1}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result text"},
            {"role": "user", "content": "next"},
        ]
        out = m._normalize_conversation_for_api(conv)
        assert [msg["role"] for msg in out] == ["user", "assistant", "user"]
        assert out[1]["content"] == [
            {"type": "text", "text": "calling"},
            {"type": "tool_use", "id": "c1", "name": "F", "input": {"x": 1}},
        ]
        # tool result and following user turn are merged into one user message
        assert out[2]["content"][0] == {
            "type": "tool_result",
            "tool_use_id": "c1",
            "content": "result text",
        }
        assert out[2]["content"][1] == {"type": "text", "text": "next"}

    def test_normalize_conversation_drops_whitespace_messages(self) -> None:
        m = make_anthropic()
        out = m._normalize_conversation_for_api(
            [{"role": "user", "content": "  "}, {"role": "user", "content": "ok"}]
        )
        assert out == [{"role": "user", "content": "ok"}]

    def test_build_create_kwargs_defaults_and_system_hoist(self) -> None:
        m = make_anthropic()
        m.conversation = [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "hi"},
        ]
        kwargs = m._build_create_kwargs()
        assert kwargs["model"] == "claude-sonnet-4-5"
        assert kwargs["system"] == "sys prompt"
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
        # sonnet-4 default: thinking enabled + boosted max_tokens
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 10000}
        assert kwargs["max_tokens"] == 64000
        assert kwargs["cache_control"] == {"type": "ephemeral"}
        assert "interleaved-thinking-2025-05-14" in kwargs["extra_headers"]["anthropic-beta"]
        assert "reasoning_effort" not in kwargs
        assert "use_responses_api" not in kwargs

    def test_build_create_kwargs_opus_adaptive(self) -> None:
        m = AnthropicModel("claude-opus-4-6", api_key="test")
        m.conversation = [{"role": "user", "content": "hi"}]
        kwargs = m._build_create_kwargs()
        assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert kwargs["max_tokens"] == 65536

    def test_build_create_kwargs_user_config_respected(self) -> None:
        m = AnthropicModel(
            "claude-sonnet-4-5",
            api_key="test",
            model_config={
                "max_tokens": 1234,
                "stop": "STOP",
                "system_instruction": "cfg sys",
                "reasoning_effort": "high",
                "use_responses_api": True,
                "enable_cache": False,
            },
        )
        m.conversation = [{"role": "user", "content": "hi"}]
        kwargs = m._build_create_kwargs(tools=[{"name": "F"}])
        assert kwargs["max_tokens"] == 1234
        assert kwargs["stop_sequences"] == ["STOP"]
        assert kwargs["system"] == "cfg sys"
        assert kwargs["tools"] == [{"name": "F"}]
        assert "cache_control" not in kwargs
        assert "reasoning_effort" not in kwargs
        assert "use_responses_api" not in kwargs

    def test_build_create_kwargs_all_whitespace_raises(self) -> None:
        m = make_anthropic()
        m.conversation = [{"role": "user", "content": "   "}]
        with pytest.raises(KISSError):
            m._build_create_kwargs()

    def test_build_anthropic_tools_schema(self) -> None:
        m = make_anthropic()
        tools = m._build_anthropic_tools_schema(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "F",
                        "description": "desc",
                        "parameters": {"type": "object", "properties": {"a": {}}},
                    },
                },
                {"type": "function", "function": {}},
            ]
        )
        assert tools == [
            {
                "name": "F",
                "description": "desc",
                "input_schema": {"type": "object", "properties": {"a": {}}},
            },
            {
                "name": "",
                "description": "",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    def test_initialize_with_attachments(self) -> None:
        m = make_anthropic()
        m.initialize("look", attachments=[Attachment(data=PNG_BYTES, mime_type="image/png")])
        assert m.conversation == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": PNG_B64,
                        },
                    },
                    {"type": "text", "text": "look"},
                ],
            }
        ]


class TestAnthropicFunctionResults:
    """add_function_results_to_conversation_and_return behavior."""

    def _model_with_tool_use(self) -> AnthropicModel:
        m = make_anthropic()
        m.conversation = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "F", "input": {}},
                    {"type": "tool_use", "id": "toolu_2", "name": "G", "input": {}},
                ],
            },
        ]
        return m

    def test_plain_string_results(self) -> None:
        m = self._model_with_tool_use()
        m.add_function_results_to_conversation_and_return(
            [("F", {"result": "out1"}), ("G", {"result": "out2"})]
        )
        assert m.conversation[-1] == {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "out1"},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "out2"},
            ],
        }

    def test_explicit_tool_use_id_and_fallback(self) -> None:
        m = make_anthropic()
        m.conversation = [{"role": "user", "content": "hi"}]
        m.add_function_results_to_conversation_and_return(
            [("F", {"result": "r", "tool_use_id": "explicit"}), ("G", {"result": "s"})]
        )
        blocks = m.conversation[-1]["content"]
        assert blocks[0]["tool_use_id"] == "explicit"
        assert blocks[1]["tool_use_id"] == "toolu_G_1"

    def test_binary_attachment_result_lifted_to_blocks(self) -> None:
        m = self._model_with_tool_use()
        payload = "Screenshot:\n" + encode_binary_attachment("image/png", PNG_BYTES)
        m.add_function_results_to_conversation_and_return([("F", {"result": payload})])
        block = m.conversation[-1]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "toolu_1"
        content_blocks = block["content"]
        assert content_blocks[0]["type"] == "text"
        assert content_blocks[0]["text"].startswith("Screenshot:")
        assert content_blocks[1] == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64},
        }

    def test_binary_pdf_and_video_attachments(self) -> None:
        m = self._model_with_tool_use()
        payload = (
            encode_binary_attachment("application/pdf", PDF_BYTES)
            + encode_binary_attachment("video/mp4", b"vid")
        )
        m.add_function_results_to_conversation_and_return([("F", {"result": payload})])
        content_blocks = m.conversation[-1]["content"][0]["content"]
        types_seen = [b["type"] for b in content_blocks]
        assert "document" in types_seen
        assert all(t in ("text", "document") for t in types_seen)

    def test_usage_info_suffix_appended(self) -> None:
        m = self._model_with_tool_use()
        m.usage_info_for_messages = "USAGE-INFO"
        m.add_function_results_to_conversation_and_return([("F", {"result": "out"})])
        content = m.conversation[-1]["content"][0]["content"]
        assert content == "out\n\nUSAGE-INFO"


class TestAnthropicTokenCounts:
    def test_no_usage_returns_zeros(self) -> None:
        m = make_anthropic()
        assert m.extract_input_output_token_counts_from_response(object()) == (0, 0, 0, 0, 0)

    def test_real_usage_object(self) -> None:
        from anthropic.types import Usage

        m = make_anthropic()
        usage = Usage(
            input_tokens=10,
            output_tokens=20,
            cache_read_input_tokens=5,
            cache_creation_input_tokens=7,
        )

        class Resp:
            """Minimal response carrier holding a real anthropic Usage."""

            def __init__(self, usage: Usage) -> None:
                self.usage = usage

        counts = m.extract_input_output_token_counts_from_response(Resp(usage))
        assert counts == (10, 20, 5, 0, 7)

    def test_get_embedding_raises(self) -> None:
        with pytest.raises(KISSError):
            make_anthropic().get_embedding("text")


class TestGeminiHelpers:
    def test_coerce_args_dict(self) -> None:
        assert _coerce_args_dict({"a": 1}) == {"a": 1}
        assert _coerce_args_dict('{"a": 1}') == {"a": 1}
        assert _coerce_args_dict("not json") == {}
        assert _coerce_args_dict("[1]") == {}
        assert _coerce_args_dict("  ") == {}
        assert _coerce_args_dict(None) == {}

    def test_decode_base64(self) -> None:
        assert _decode_base64(PNG_B64) == PNG_BYTES
        assert _decode_base64("A") is None  # invalid length raises binascii.Error

    def test_media_block_anthropic_image(self) -> None:
        part = _media_block_to_part(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64},
            }
        )
        assert part is not None
        assert part.inline_data is not None
        assert part.inline_data.data == PNG_BYTES
        assert part.inline_data.mime_type == "image/png"

    def test_media_block_openai_image_url(self) -> None:
        part = _media_block_to_part(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{PNG_B64}"}}
        )
        assert part is not None
        assert part.inline_data is not None
        assert part.inline_data.mime_type == "image/jpeg"

    def test_media_block_openai_image_url_default_mime(self) -> None:
        part = _media_block_to_part(
            {"type": "image_url", "image_url": {"url": f"data:;base64,{PNG_B64}"}}
        )
        assert part is not None
        assert part.inline_data is not None
        assert part.inline_data.mime_type == "image/png"

    def test_media_block_openai_file(self) -> None:
        part = _media_block_to_part(
            {"type": "file", "file": {"file_data": f"data:application/pdf;base64,{PDF_B64}"}}
        )
        assert part is not None
        assert part.inline_data is not None
        assert part.inline_data.mime_type == "application/pdf"

    def test_media_block_remote_url_dropped(self) -> None:
        assert (
            _media_block_to_part(
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
            )
            is None
        )

    def test_media_block_bad_base64_dropped(self) -> None:
        assert (
            _media_block_to_part(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "A"},
                }
            )
            is None
        )

    def test_tool_result_response_dict(self) -> None:
        assert _tool_result_response_dict('{"a": 1}') == {"a": 1}
        assert _tool_result_response_dict("[1, 2]") == {"result": [1, 2]}
        assert _tool_result_response_dict("plain") == {"result": "plain"}
        assert _tool_result_response_dict(
            [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]
        ) == {"result": "xy"}
        assert _tool_result_response_dict(42) == {"result": 42}


class TestGeminiConversion:
    def make_gemini(self) -> GeminiModel:
        return GeminiModel("gemini-2.5-pro", api_key="test")

    def test_convert_conversation_roles_and_tools(self) -> None:
        m = self.make_gemini()
        m.conversation = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "text",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "F", "arguments": '{"x": 1}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'},
        ]
        contents = m._convert_conversation_to_gemini_contents()
        assert [c.role for c in contents] == ["user", "model", "user"]
        model_parts = contents[1].parts
        assert model_parts is not None
        assert model_parts[0].text == "text"
        assert model_parts[1].function_call is not None
        assert model_parts[1].function_call.name == "F"
        assert model_parts[1].function_call.args == {"x": 1}
        tool_parts = contents[2].parts
        assert tool_parts is not None
        assert tool_parts[0].function_response is not None
        assert tool_parts[0].function_response.name == "F"
        assert tool_parts[0].function_response.response == {"ok": True}

    def test_convert_user_attachments(self) -> None:
        m = self.make_gemini()
        m.conversation = [
            {
                "role": "user",
                "content": "look",
                "attachments": [Attachment(data=PNG_BYTES, mime_type="image/png")],
            }
        ]
        contents = m._convert_conversation_to_gemini_contents()
        parts = contents[0].parts
        assert parts is not None
        assert parts[0].inline_data is not None
        assert parts[0].inline_data.data == PNG_BYTES
        assert parts[1].text == "look"

    def test_convert_anthropic_block_list(self) -> None:
        m = self.make_gemini()
        m.conversation = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "secret"},
                    {"type": "text", "text": "visible"},
                    {"type": "tool_use", "id": "t1", "name": "G", "input": {"a": 2}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "done"}
                ],
            },
        ]
        contents = m._convert_conversation_to_gemini_contents()
        model_parts = contents[0].parts
        assert model_parts is not None
        assert [p.text for p in model_parts if p.text] == ["visible"]
        assert model_parts[-1].function_call is not None
        user_parts = contents[1].parts
        assert user_parts is not None
        assert user_parts[0].function_response is not None
        assert user_parts[0].function_response.name == "G"

    def test_add_function_results_lifts_attachments(self) -> None:
        m = self.make_gemini()
        m.conversation = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c9",
                        "type": "function",
                        "function": {"name": "Read", "arguments": "{}"},
                    }
                ],
            },
        ]
        payload = "img:\n" + encode_binary_attachment("image/png", PNG_BYTES)
        m.add_function_results_to_conversation_and_return([("Read", {"result": payload})])
        tool_msg = m.conversation[2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "c9"
        assert "KISS_BINARY_ATTACHMENT" not in tool_msg["content"]
        follow_up = m.conversation[3]
        assert follow_up["role"] == "user"
        assert follow_up["attachments"][0].data == PNG_BYTES

    def test_resolve_system_instruction_merges(self) -> None:
        m = GeminiModel(
            "gemini-2.5-pro", api_key="test", model_config={"system_instruction": "cfg"}
        )
        m.conversation = [
            {"role": "system", "content": "sys1"},
            {"role": "system", "content": [{"type": "text", "text": "sys2"}]},
            {"role": "system", "content": "cfg"},  # duplicate skipped
            {"role": "user", "content": "hi"},
        ]
        assert m._resolve_system_instruction() == "cfg\n\nsys1\n\nsys2"

    def test_build_config(self) -> None:
        m = GeminiModel(
            "gemini-2.5-pro",
            api_key="test",
            model_config={"max_tokens": 100, "temperature": 0.5, "top_p": 0.9},
        )
        m.conversation = [{"role": "user", "content": "hi"}]
        config = m._build_config()
        assert config.max_output_tokens == 100
        assert config.temperature == 0.5
        assert config.top_p == 0.9
        assert config.thinking_config is not None
        assert config.thinking_config.include_thoughts is True

    def test_reset_conversation_clears_thought_signatures(self) -> None:
        m = self.make_gemini()
        m._thought_signatures["x"] = b"sig"
        m.reset_conversation()
        assert m._thought_signatures == {}

    def test_token_counts_real_response(self) -> None:
        m = self.make_gemini()
        response = gtypes.GenerateContentResponse(
            usage_metadata=gtypes.GenerateContentResponseUsageMetadata(
                prompt_token_count=100,
                candidates_token_count=20,
                thoughts_token_count=5,
                cached_content_token_count=30,
            )
        )
        assert m.extract_input_output_token_counts_from_response(response) == (70, 25, 30, 0)

    def test_token_counts_no_usage(self) -> None:
        m = self.make_gemini()
        assert m.extract_input_output_token_counts_from_response(object()) == (0, 0, 0, 0)


class TestCodexModel:
    def make_codex(self) -> CodexModel:
        return CodexModel("codex/gpt-5-codex")

    def test_cli_model_stripping(self) -> None:
        assert self.make_codex()._cli_model == "gpt-5-codex"
        assert CodexModel("codex/default")._cli_model == "default"
        assert CodexModel("plain")._cli_model == "plain"

    def test_build_prompt_single_turn(self) -> None:
        m = self.make_codex()
        m.initialize("just this")
        assert m._build_prompt() == "just this"

    def test_build_prompt_multi_turn_with_system(self) -> None:
        m = CodexModel("codex/gpt-5-codex", model_config={"system_instruction": "SYS"})
        m.initialize("q1")
        m.conversation.append({"role": "assistant", "content": "a1"})
        m.conversation.append({"role": "tool", "content": "t1"})
        assert m._build_prompt() == (
            "[System]: SYS\n\n[User]: q1\n\n[Assistant]: a1\n\n[Tool Result]: t1"
        )

    def test_build_prompt_single_turn_with_system(self) -> None:
        m = CodexModel("codex/gpt-5-codex", model_config={"system_instruction": "SYS"})
        m.initialize("q1")
        assert m._build_prompt() == "[System]: SYS\n\n[User]: q1"

    def test_find_in_candidate_paths(self, tmp_path) -> None:
        exe = tmp_path / "codex"
        exe.write_bytes(b"#!/bin/sh\n")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        missing = tmp_path / "missing"
        assert _find_in_candidate_paths([str(missing), str(exe)]) == str(exe)
        assert _find_in_candidate_paths([str(missing)]) is None
        assert _find_in_candidate_paths([]) is None

    def test_parse_stream_events_happy_path(self) -> None:
        m = self.make_codex()
        tokens: list[str] = []
        thinking: list[bool] = []
        def on_token(token: str) -> None:
            tokens.append(token)

        def on_thinking(is_start: bool) -> None:
            thinking.append(is_start)

        m.token_callback = on_token
        m.thinking_callback = on_thinking
        events = [
            {"type": "thread.started", "thread_id": "th_1"},
            {"type": "turn.started"},
            {"type": "item.started", "item": {"type": "command_execution", "command": "ls"}},
            {
                "type": "item.completed",
                "item": {"type": "command_execution", "aggregated_output": "file.txt"},
            },
            {"type": "item.completed", "item": {"type": "agent_reasoning", "text": "hmm"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "Hello"}},
            {"type": "turn.completed", "usage": {"input_tokens": 3, "output_tokens": 4}},
        ]
        lines = [json.dumps(e) for e in events] + ["", "not json"]
        content, result, err = m._parse_stream_events(iter(lines))
        assert content == "Hello"
        assert err is None
        assert result == {
            "thread_id": "th_1",
            "usage": {"input_tokens": 3, "output_tokens": 4},
        }
        assert "Hello" in tokens
        assert "$ ls\n" in tokens
        assert "file.txt" in tokens
        assert "hmm" in tokens
        assert thinking == [True, False, True, False, True, False]

    def test_parse_stream_events_text_and_thinking_deltas(self) -> None:
        m = self.make_codex()
        events = [
            {"type": "text_delta", "delta": {"type": "text_delta", "text": "Hi "}},
            {"type": "text_delta", "delta": {"type": "text_delta", "text": "there"}},
            {"type": "thinking_delta", "delta": {"type": "thinking_delta", "text": "T"}},
            {"type": "thinking_start"},
            {"type": "thinking_end"},
        ]
        content, _result, err = m._parse_stream_events(json.dumps(e) for e in events)
        assert content == "Hi there"
        assert err is None

    def test_parse_stream_events_error(self) -> None:
        m = self.make_codex()
        events = [{"type": "error", "message": "boom"}]
        _c, _r, err = m._parse_stream_events(json.dumps(e) for e in events)
        assert err == "boom"

    def test_parse_stream_events_turn_failed(self) -> None:
        m = self.make_codex()
        events = [{"type": "turn.failed", "error": {"message": "bad turn"}}]
        _c, _r, err = m._parse_stream_events(json.dumps(e) for e in events)
        assert err == "bad turn"

    def test_token_counts(self) -> None:
        m = self.make_codex()
        response = {
            "usage": {"input_tokens": 100, "cached_input_tokens": 30, "output_tokens": 7}
        }
        assert m.extract_input_output_token_counts_from_response(response) == (70, 7, 30, 0)
        assert m.extract_input_output_token_counts_from_response("nope") == (0, 0, 0, 0)

    def test_get_embedding_raises(self) -> None:
        with pytest.raises(KISSError):
            self.make_codex().get_embedding("x")


class TestClaudeCodeModel:
    def make_cc(self) -> ClaudeCodeModel:
        return ClaudeCodeModel("cc/sonnet")

    def test_cache_creation_tokens(self) -> None:
        assert _claude_code_cache_creation_tokens(
            {
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 3,
                    "ephemeral_1h_input_tokens": 4,
                }
            }
        ) == (3, 4)
        assert _claude_code_cache_creation_tokens({"cache_creation_input_tokens": 9}) == (0, 9)
        assert _claude_code_cache_creation_tokens({}) == (0, 0)

    def test_find_consecutive_tool_calls_end(self) -> None:
        one = '{"tool_calls": [{"name": "Bash", "arguments": {}}]}'
        assert _find_consecutive_tool_calls_end(one) == len(one)
        two = one + "  \n" + one
        assert _find_consecutive_tool_calls_end(two) == len(two)
        interrupted = one + " (no output) " + one
        assert _find_consecutive_tool_calls_end(interrupted) == len(one)
        assert _find_consecutive_tool_calls_end("no json here") == -1

    def test_build_prompt(self) -> None:
        m = self.make_cc()
        m.initialize("only")
        assert m._build_prompt() == "only"
        m.conversation.append({"role": "assistant", "content": "ans"})
        m.conversation.append({"role": "tool", "content": "res"})
        assert m._build_prompt() == "[User]: only\n\n[Assistant]: ans\n\n[Tool Result]: res"

    def test_parse_stream_events_result_authoritative(self) -> None:
        m = self.make_cc()
        events = [
            {
                "type": "assistant",
                "message": {"id": "m1", "content": [{"type": "text", "text": "partial"}]},
            },
            {"type": "result", "result": "final", "usage": {"input_tokens": 1}},
        ]
        content, result = m._parse_stream_events(json.dumps(e) for e in events)
        assert content == "final"
        assert result["usage"] == {"input_tokens": 1}
        assert m._pre_result_content == "partial"

    def test_parse_stream_events_second_assistant_stops(self) -> None:
        m = self.make_cc()
        events = [
            {
                "type": "assistant",
                "message": {"id": "m1", "content": [{"type": "text", "text": "first"}]},
            },
            {
                "type": "assistant",
                "message": {"id": "m2", "content": [{"type": "text", "text": "second"}]},
            },
        ]
        content, _ = m._parse_stream_events(json.dumps(e) for e in events)
        assert content == "first"

    def test_parse_stream_events_deltas_and_thinking(self) -> None:
        m = self.make_cc()
        tokens: list[str] = []
        thinking: list[bool] = []
        def on_token(token: str) -> None:
            tokens.append(token)

        def on_thinking(is_start: bool) -> None:
            thinking.append(is_start)

        m.token_callback = on_token
        m.thinking_callback = on_thinking
        events = [
            {"type": "content_block_start", "content_block": {"type": "thinking"}},
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "deep"},
                },
            },
            {"type": "content_block_stop"},
            {"type": "content_block_start", "content_block": {"type": "text"}},
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "answer"},
            },
            {"type": "content_block_stop"},
        ]
        content, _ = m._parse_stream_events(json.dumps(e) for e in events)
        assert content == "answer"
        assert m._last_thinking_content == "deep"
        assert tokens == ["deep", "answer"]
        assert thinking == [True, False]

    def test_parse_stream_events_stop_on_tool_calls_with_trailing(self) -> None:
        m = self.make_cc()
        tc = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        events = [
            {"type": "content_block_start", "content_block": {"type": "text"}},
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": tc},
            },
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": " hallucinated output"},
            },
            {"type": "result", "result": "SHOULD NOT BE USED"},
        ]
        content, result = m._parse_stream_events(
            (json.dumps(e) for e in events), stop_on_tool_calls=True
        )
        assert content == tc
        assert m._stopped_for_tool_calls is True
        # result event is still scanned after the early stop for usage data
        assert result.get("result") == "SHOULD NOT BE USED"

    def test_parse_stream_events_tool_calls_clean_finish(self) -> None:
        m = self.make_cc()
        tc = '{"tool_calls": [{"name": "F", "arguments": {}}]}'
        events = [
            {"type": "content_block_start", "content_block": {"type": "text"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": tc}},
            {"type": "result", "result": "", "usage": {"output_tokens": 5}},
        ]
        content, result = m._parse_stream_events(
            (json.dumps(e) for e in events), stop_on_tool_calls=True
        )
        assert content == tc
        assert m._stopped_for_tool_calls is True
        assert result["usage"] == {"output_tokens": 5}

    def test_token_counts(self) -> None:
        m = self.make_cc()
        response = {
            "usage": {
                "input_tokens": 11,
                "output_tokens": 22,
                "cache_read_input_tokens": 33,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 44,
                    "ephemeral_1h_input_tokens": 55,
                },
            }
        }
        assert m.extract_input_output_token_counts_from_response(response) == (
            11,
            22,
            33,
            44,
            55,
        )
        assert m.extract_input_output_token_counts_from_response(None) == (0, 0, 0, 0, 0)

    def test_get_embedding_raises(self) -> None:
        with pytest.raises(KISSError):
            self.make_cc().get_embedding("x")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([os.path.abspath(__file__), "-v", "-p", "no:cov"])
