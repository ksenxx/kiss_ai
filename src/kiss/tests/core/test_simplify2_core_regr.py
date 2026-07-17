# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for the second core simplification pass.

Locks down behavior that must be identical before and after:

1. The OpenRouter/DeepSeek helper trio shared by ``OpenAICompatibleModel``
   (Chat Completions) and ``OpenAICompatibleModel2`` (Responses):
   ``_is_deepseek_reasoning_model``, ``_is_openrouter_anthropic`` and
   ``_apply_cache_control_for_openrouter_anthropic`` must behave
   identically on both classes.
2. The established ``model_info`` API, including flaky-model metadata
   and the expensive-model picker.  These importable public symbols must
   remain available even when there are no in-tree production callers.
3. The stateless-CLI transport behavior shared by ``ClaudeCodeModel`` and
   ``CodexModel``: ``initialize``, single-turn and multi-turn prompt
   flattening (``_build_prompt``), and the always-raising
   ``get_embedding``.

No mocks, patches, fakes, or monkeypatching: every test constructs real
model objects.  No network calls are made (``base_url`` is never
contacted by the helpers under test).
"""

from __future__ import annotations

from typing import Any

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.codex_model import CodexModel
from kiss.core.models.model import Attachment
from kiss.core.models.model_info import (
    FLAKY_MODELS,
    MODEL_INFO,
    calculate_cost,
    get_fallback_model,
    get_flaky_reason,
    get_max_context_length,
    get_most_expensive_model,
    is_model_flaky,
    model,
)
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2

BASE_URL = "http://localhost:1"
API_KEY = "test"

MODEL_CLASSES = (OpenAICompatibleModel, OpenAICompatibleModel2)


def make(cls: type, model_name: str, model_config: dict[str, Any] | None = None) -> Any:
    """Construct a v1 or v2 OpenAI-compatible model pointing at a dead endpoint."""
    return cls(
        model_name=model_name,
        base_url=BASE_URL,
        api_key=API_KEY,
        model_config=model_config,
    )


def test_is_openrouter_anthropic_parity() -> None:
    for cls in MODEL_CLASSES:
        assert make(cls, "openrouter/anthropic/claude-opus-4.7")._is_openrouter_anthropic()
        assert not make(cls, "openrouter/deepseek/deepseek-r1")._is_openrouter_anthropic()
        assert not make(cls, "gpt-4o")._is_openrouter_anthropic()


def test_is_deepseek_reasoning_model_parity() -> None:
    for cls in MODEL_CLASSES:
        assert make(cls, "deepseek-ai/DeepSeek-R1")._is_deepseek_reasoning_model()
        # The openrouter/ routing prefix is stripped before matching.
        assert make(cls, "openrouter/deepseek/deepseek-r1")._is_deepseek_reasoning_model()
        assert not make(cls, "gpt-4o")._is_deepseek_reasoning_model()


def test_cache_control_applied_for_openrouter_anthropic_parity() -> None:
    for cls in MODEL_CLASSES:
        m = make(cls, "openrouter/anthropic/claude-opus-4.7")
        kwargs: dict[str, Any] = {}
        m._apply_cache_control_for_openrouter_anthropic(kwargs)
        assert kwargs["extra_body"]["cache_control"] == {"type": "ephemeral"}


def test_cache_control_noop_when_disabled_or_other_vendor_parity() -> None:
    for cls in MODEL_CLASSES:
        m = make(
            cls,
            "openrouter/anthropic/claude-opus-4.7",
            model_config={"enable_cache": False},
        )
        kwargs: dict[str, Any] = {}
        m._apply_cache_control_for_openrouter_anthropic(kwargs)
        assert kwargs == {}

        m2 = make(cls, "gpt-4o")
        kwargs2: dict[str, Any] = {}
        m2._apply_cache_control_for_openrouter_anthropic(kwargs2)
        assert kwargs2 == {}


def test_cache_control_copies_caller_extra_body_parity() -> None:
    """The caller-supplied ``extra_body`` dict must never be mutated in place."""
    for cls in MODEL_CLASSES:
        m = make(cls, "openrouter/anthropic/claude-opus-4.7")
        caller_extra_body: dict[str, Any] = {"provider": {"order": ["anthropic"]}}
        kwargs: dict[str, Any] = {"extra_body": caller_extra_body}
        m._apply_cache_control_for_openrouter_anthropic(kwargs)
        assert kwargs["extra_body"]["cache_control"] == {"type": "ephemeral"}
        assert kwargs["extra_body"]["provider"] == {"order": ["anthropic"]}
        assert "cache_control" not in caller_extra_body


def test_model_factory_routes_base_url_override_to_v1() -> None:
    m = model(
        "totally-custom-model",
        model_config={"base_url": BASE_URL, "api_key": API_KEY},
    )
    assert isinstance(m, OpenAICompatibleModel)
    assert m.model_name == "totally-custom-model"


def test_model_info_public_surface_is_preserved() -> None:
    flaky = "openrouter/baidu/ernie-4.5-21b-a3b"
    assert FLAKY_MODELS[flaky] == "Ignores function calling tools"
    assert is_model_flaky(flaky)
    assert get_flaky_reason(flaky) == "Ignores function calling tools"
    assert not is_model_flaky("unknown-model-xyz")
    assert get_flaky_reason("unknown-model-xyz") == ""
    assert isinstance(get_most_expensive_model(), str)
    assert isinstance(get_most_expensive_model(fc_only=False), str)

    assert "claude-fable-5" in MODEL_INFO
    assert get_max_context_length("claude-fable-5") > 0
    assert calculate_cost("claude-sonnet-5", 1_000_000, 0) > 0.0
    assert calculate_cost("unknown-model-xyz", 0, 0) == 0.0
    assert get_fallback_model("unknown-model-xyz") is None


CLI_MODEL_CLASSES = (ClaudeCodeModel, CodexModel)


def make_cli(cls: type, model_config: dict[str, Any] | None = None) -> Any:
    """Construct a CLI-backed model (no subprocess is spawned before generate())."""
    name = "cc/opus" if cls is ClaudeCodeModel else "codex/default"
    return cls(model_name=name, model_config=model_config)


def test_cli_models_initialize_sets_single_user_message() -> None:
    for cls in CLI_MODEL_CLASSES:
        m = make_cli(cls)
        m.initialize("hello world")
        assert m.conversation == [{"role": "user", "content": "hello world"}]


def test_cli_models_single_turn_prompt_is_bare_text() -> None:
    for cls in CLI_MODEL_CLASSES:
        m = make_cli(cls)
        m.initialize("solve the task")
        assert m._build_prompt() == "solve the task"


def test_cli_models_multi_turn_prompt_renders_dialogue() -> None:
    for cls in CLI_MODEL_CLASSES:
        m = make_cli(cls)
        m.initialize("first question")
        m.add_message_to_conversation("assistant", "an answer")
        m.conversation.append(
            {"role": "tool", "tool_call_id": "call_1", "content": "tool output"}
        )
        prompt = m._build_prompt()
        assert "[User]: first question" in prompt
        assert "[Assistant]: an answer" in prompt
        assert "[Tool Result]: tool output" in prompt


def test_codex_prompt_prepends_system_instruction() -> None:
    m = make_cli(CodexModel, model_config={"system_instruction": "be terse"})
    m.initialize("do it")
    prompt = m._build_prompt()
    assert prompt.startswith("[System]: be terse")
    assert "[User]: do it" in prompt


def test_cli_models_get_embedding_raises() -> None:
    for cls in CLI_MODEL_CLASSES:
        m = make_cli(cls)
        with pytest.raises(KISSError, match="does not support embeddings"):
            m.get_embedding("text")


def test_cli_model_subclasses_keep_transport_error_names() -> None:
    class DerivedClaude(ClaudeCodeModel):
        pass

    class DerivedCodex(CodexModel):
        pass

    for cls, name, expected in (
        (DerivedClaude, "cc/opus", "ClaudeCodeModel"),
        (DerivedCodex, "codex/default", "CodexModel"),
    ):
        m = cls(model_name=name)
        with pytest.raises(KISSError) as exc_info:
            m.get_embedding("text")
        assert exc_info.value.args == (f"{expected} does not support embeddings.",)


def test_cli_attachment_warning_preserves_logger_and_record(caplog: Any) -> None:
    attachment = Attachment(data=b"x", mime_type="image/png")
    for cls, logger_name, expected in (
        (
            ClaudeCodeModel,
            "kiss.core.models.claude_code_model",
            "ClaudeCodeModel does not support attachments; they will be ignored.",
        ),
        (
            CodexModel,
            "kiss.core.models.codex_model",
            "CodexModel does not support attachments; they will be ignored.",
        ),
    ):
        caplog.clear()
        make_cli(cls).initialize("hello", [attachment])
        record = caplog.records[-1]
        assert record.name == logger_name
        assert record.msg == expected
        assert record.args == ()
