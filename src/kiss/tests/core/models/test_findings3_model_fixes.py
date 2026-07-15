# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for the findings-3.md model-layer fixes.

Each test exercises real model objects through their actual code paths —
no mocks, patches, or fakes.  Bug-focused cases fail before their fixes;
additional preservation cases pin unchanged precedence and alias behavior:

* #3  — v1 OpenRouter-Anthropic ``cache_control`` must not mutate the
        caller's nested ``extra_body`` dict.
* #4  — v2 must strip the v1-only ``use_responses_api`` config key
        instead of forwarding it to ``client.responses.create``.
* #5  — v1 must not inject the MODEL_INFO default ``reasoning_effort``
        when the caller passed a native ``reasoning={"effort": ...}``.
* #7  — non-string (dict/list) tool results must be JSON-encoded, not
        crash ``parse_binary_attachments``.
* #18 — ``get_max_context_length`` raises ``KISSError`` (not
        ``KeyError``) for unknown models, matching ``calculate_cost``.
* #22 — Gemini honors ``max_completion_tokens`` via
        ``max_output_tokens`` like v1/v2/Anthropic.
* #1  — the ``-xhigh`` alias stripping shared with ``model_info`` still
        yields the same wire model names.
"""

from __future__ import annotations

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.model_info import get_max_context_length
from kiss.core.models.openai_compatible_model import (
    OpenAICompatibleModel,
    _provider_model_name,
)
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2

_ASSISTANT_TOOL_CALL_MSG = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "f", "arguments": "{}"},
        }
    ],
}


class TestFinding3CacheControlNoCallerMutation:
    """#3 — v1 cache_control must not leak into the caller's config."""

    def test_v1_build_chat_kwargs_does_not_mutate_caller_extra_body(self) -> None:
        caller_config = {"extra_body": {"custom": 1}}
        m = OpenAICompatibleModel(
            "openrouter/anthropic/claude-fable-5",
            base_url="https://openrouter.ai/api/v1",
            api_key="k",
            model_config=caller_config,
        )
        kwargs = m._build_chat_kwargs([{"role": "user", "content": "hi"}])
        # The request kwargs DO carry the injected cache marker...
        assert kwargs["extra_body"]["cache_control"] == {"type": "ephemeral"}
        assert kwargs["extra_body"]["custom"] == 1
        # ...but neither the caller's dict nor the stored config was mutated.
        assert caller_config["extra_body"] == {"custom": 1}
        assert m.model_config["extra_body"] == {"custom": 1}

    def test_v1_second_request_sees_clean_config(self) -> None:
        m = OpenAICompatibleModel(
            "openrouter/anthropic/claude-fable-5",
            base_url="https://openrouter.ai/api/v1",
            api_key="k",
            model_config={"extra_body": {"custom": 1}},
        )
        m._build_chat_kwargs([{"role": "user", "content": "hi"}])
        assert "cache_control" not in m.model_config["extra_body"]


class TestFinding4UseResponsesApiStripped:
    """#4 — v2 must drop the v1-only ``use_responses_api`` key."""

    def test_v2_shape_responses_kwargs_pops_use_responses_api(self) -> None:
        m = OpenAICompatibleModel2(
            "gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key="k",
            model_config={"use_responses_api": True, "temperature": 0.5},
        )
        kwargs = m._shape_responses_kwargs(
            input_items=[
                {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
            ],
            tools=None,
        )
        assert "use_responses_api" not in kwargs
        assert kwargs["temperature"] == 0.5


class TestFinding5NativeReasoningDictWins:
    """#5 — a caller-native ``reasoning`` dict suppresses the default."""

    def test_v1_skips_default_when_native_reasoning_effort_present(self) -> None:
        # gpt-5.5 declares thinking="high" in MODEL_INFO.
        m = OpenAICompatibleModel(
            "gpt-5.5",
            base_url="https://api.openai.com/v1",
            api_key="k",
            model_config={"reasoning": {"effort": "low"}},
        )
        assert "reasoning_effort" not in m.model_config
        assert m.model_config["reasoning"] == {"effort": "low"}

    def test_v1_still_defaults_without_native_reasoning(self) -> None:
        m = OpenAICompatibleModel(
            "gpt-5.5",
            base_url="https://api.openai.com/v1",
            api_key="k",
        )
        assert m.model_config["reasoning_effort"] == "high"

    def test_v1_matches_v2_for_native_reasoning_config(self) -> None:
        config = {"reasoning": {"effort": "low"}}
        v1 = OpenAICompatibleModel(
            "gpt-5.5", base_url="https://api.openai.com/v1", api_key="k",
            model_config=dict(config),
        )
        v2 = OpenAICompatibleModel2(
            "gpt-5.5", base_url="https://api.openai.com/v1", api_key="k",
            model_config=dict(config),
        )
        assert ("reasoning_effort" in v1.model_config) == (
            "reasoning_effort" in v2.model_config
        )


class TestFinding7NonStringToolResults:
    """#7 — dict/list tool results are JSON-encoded, never a crash."""

    def test_v1_dict_tool_result_is_json_encoded(self) -> None:
        m = OpenAICompatibleModel(
            "gpt-4o", base_url="https://api.openai.com/v1", api_key="k"
        )
        m.conversation = [
            {"role": "user", "content": "hi"},
            dict(_ASSISTANT_TOOL_CALL_MSG),
        ]
        m.add_function_results_to_conversation_and_return(
            [("f", {"result": {"a": 1, "b": [2, 3]}})]
        )
        tool_msg = m.conversation[-1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["content"] == '{"a": 1, "b": [2, 3]}'

    def test_base_model_dict_tool_result_is_json_encoded(self) -> None:
        # ClaudeCodeModel inherits the base Model implementation.
        m = ClaudeCodeModel("claude-fable-5")
        m.conversation = [
            {"role": "user", "content": "hi"},
            dict(_ASSISTANT_TOOL_CALL_MSG),
        ]
        m.add_function_results_to_conversation_and_return(
            [("f", {"result": {"ключ": "значение"}})]
        )
        tool_msg = m.conversation[-1]
        assert tool_msg["role"] == "tool"
        # ensure_ascii=False keeps non-ASCII text readable.
        assert tool_msg["content"] == '{"ключ": "значение"}'

    def test_v1_list_tool_result_does_not_crash(self) -> None:
        m = OpenAICompatibleModel(
            "gpt-4o", base_url="https://api.openai.com/v1", api_key="k"
        )
        m.conversation = [
            {"role": "user", "content": "hi"},
            dict(_ASSISTANT_TOOL_CALL_MSG),
        ]
        m.add_function_results_to_conversation_and_return(
            [("f", {"result": [1, "two", None]})]
        )
        assert m.conversation[-1]["content"] == '[1, "two", null]'


class TestFinding18UnknownModelErrorType:
    """#18 — unknown models raise KISSError from get_max_context_length."""

    def test_get_max_context_length_raises_kiss_error(self) -> None:
        with pytest.raises(KISSError):
            get_max_context_length("no-such-model-xyz-12345")

    def test_get_max_context_length_known_model_still_works(self) -> None:
        assert get_max_context_length("gpt-4o") > 0


class TestFinding22GeminiMaxCompletionTokens:
    """#22 — Gemini honors ``max_completion_tokens``."""

    def test_build_config_uses_max_completion_tokens(self) -> None:
        m = GeminiModel(
            "gemini-2.5-pro",
            api_key="k",
            model_config={"max_completion_tokens": 123},
        )
        m.conversation = []
        cfg = m._build_config()
        assert cfg.max_output_tokens == 123

    def test_build_config_max_tokens_takes_precedence(self) -> None:
        m = GeminiModel(
            "gemini-2.5-pro",
            api_key="k",
            model_config={"max_tokens": 77, "max_completion_tokens": 123},
        )
        m.conversation = []
        cfg = m._build_config()
        assert cfg.max_output_tokens == 77

    def test_explicit_zero_max_tokens_still_takes_precedence(self) -> None:
        """Precedence is based on presence/None, matching Anthropic."""
        m = GeminiModel(
            "gemini-2.5-pro",
            api_key="k",
            model_config={"max_tokens": 0, "max_completion_tokens": 123},
        )
        cfg = m._build_config()
        assert cfg.max_output_tokens == 0


class TestFinding1XhighAliasSharedHelper:
    """#1 — the shared -xhigh stripping keeps the same wire names."""

    def test_xhigh_alias_stripped_from_api_model_name(self) -> None:
        m = OpenAICompatibleModel(
            "gpt-5.5-xhigh", base_url="https://api.openai.com/v1", api_key="k"
        )
        assert m._api_model_name == "gpt-5.5"

    def test_openrouter_prefix_and_plain_names(self) -> None:
        assert (
            _provider_model_name("openrouter/anthropic/claude-fable-5")
            == "anthropic/claude-fable-5"
        )
        assert _provider_model_name("gpt-4o") == "gpt-4o"
        assert (
            _provider_model_name("openrouter/openai/gpt-5.5-xhigh")
            == "openai/gpt-5.5"
        )
