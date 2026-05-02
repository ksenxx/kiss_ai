"""Integration tests targeting uncovered branches. No mocks, patches, or test doubles."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import TestCase

import pytest


class TestModelBareListSchema(TestCase):
    def test_bare_list_type_produces_array(self) -> None:
        """list (no type args) should produce {"type": "array"}."""
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

        m = OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        import typing
        schema = m._python_type_to_json_schema(typing.List)  # noqa: UP006
        assert schema == {"type": "array"}


class TestDeepSeekReasoningNoMatch(TestCase):
    def test_no_think_tags(self) -> None:
        from kiss.core.models.openai_compatible_model import (
            _extract_deepseek_reasoning,
        )

        reasoning, answer = _extract_deepseek_reasoning("Just a plain answer")
        assert reasoning == ""
        assert answer == "Just a plain answer"

class TestCacheControlOpenRouter(TestCase):
    def test_openrouter_anthropic_adds_cache(self) -> None:
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

        m = OpenAICompatibleModel(
            model_name="openrouter/anthropic/claude-3.5-sonnet",
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
        )
        kwargs: dict[str, Any] = {}
        m._apply_cache_control_for_openrouter_anthropic(kwargs)
        assert kwargs["extra_body"]["cache_control"] == {"type": "ephemeral"}

    def test_openrouter_cache_disabled(self) -> None:
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

        m = OpenAICompatibleModel(
            model_name="openrouter/anthropic/claude-3.5-sonnet",
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
            model_config={"enable_cache": False},
        )
        kwargs: dict[str, Any] = {}
        m._apply_cache_control_for_openrouter_anthropic(kwargs)
        assert "extra_body" not in kwargs


class TestTextBasedToolsParsing(TestCase):
    def test_parse_text_based_tool_calls_invalid_json(self) -> None:
        from kiss.core.models.model import (
            _parse_text_based_tool_calls,
        )

        content = '```json\n{broken json}\n```'
        calls = _parse_text_based_tool_calls(content)
        assert calls == []


class TestModelStr(TestCase):
    def test_model_str_gemini(self) -> None:
        """GeminiModel inherits Model.__str__ (line 172) — no override."""
        from kiss.core.models.gemini_model import GeminiModel

        m = GeminiModel(model_name="gemini-2.0-flash", api_key="test-key")
        s = str(m)
        assert "GeminiModel" in s
        assert "gemini-2.0-flash" in s
        r = repr(m)
        assert r == s


class TestAnthropicStopValNotStrOrList(TestCase):
    def test_stop_val_integer_ignored(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel(
            model_name="claude-3-haiku-20240307",
            api_key="test-key",
            model_config={"stop": 42},
        )
        kwargs = m._build_create_kwargs()
        assert "stop_sequences" not in kwargs
        assert "stop" not in kwargs


class TestAnthropicAddFunctionResultsNoToolUse(TestCase):
    def test_no_tool_use_blocks_in_content(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel(
            model_name="claude-3-haiku-20240307",
            api_key="test-key",
        )
        m.conversation = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]
        m.add_function_results_to_conversation_and_return(
            [("func1", {"result": "ok"})]
        )
        last = m.conversation[-1]
        assert last["role"] == "user"
        assert len(last["content"]) == 1
        assert last["content"][0]["tool_use_id"] == "toolu_func1_0"


class TestGeminiNonStrContent(TestCase):
    def test_non_str_user_content_skipped(self) -> None:
        from kiss.core.models.gemini_model import GeminiModel

        m = GeminiModel(
            model_name="gemini-2.0-flash",
            api_key="test-key",
        )
        m.initialize("hello")
        m.conversation[0]["content"] = 12345
        contents = m._convert_conversation_to_gemini_contents()
        assert len(contents) == 0


class TestGeminiBuildConfigThinkingProvided(TestCase):
    def test_custom_thinking_config(self) -> None:
        from google.genai import types

        from kiss.core.models.gemini_model import GeminiModel

        custom_tc = types.ThinkingConfig(include_thoughts=False)
        m = GeminiModel(
            model_name="gemini-2.0-flash",
            api_key="test-key",
            model_config={"thinking_config": custom_tc},
        )
        m.initialize("hello")
        config = m._build_config()
        assert config.thinking_config == custom_tc


class TestGeminiPartsFromResponseEmpty(TestCase):
    def test_none_response(self) -> None:
        from kiss.core.models.gemini_model import GeminiModel

        assert GeminiModel._parts_from_response(None) == []


class TestModelsInit:
    def test_models_import_succeeds(self) -> None:
        from kiss.core.models import (
            AnthropicModel,
            Attachment,
            GeminiModel,
            Model,
            OpenAICompatibleModel,
        )

        assert AnthropicModel is not None
        assert OpenAICompatibleModel is not None
        assert GeminiModel is not None
        assert Model is not None
        assert Attachment is not None


class TestModelInfoFactory:
    def test_model_openrouter(self) -> None:
        from kiss.core.models.model_info import model

        m = model("openrouter/foo-bar")
        assert m.model_name == "openrouter/foo-bar"

    def test_model_together_prefix(self) -> None:
        from kiss.core.models.model_info import model

        m = model("meta-llama/Llama-3.3-70B-Instruct-Turbo")
        assert m.model_name == "meta-llama/Llama-3.3-70B-Instruct-Turbo"

    def test_model_text_embedding_004(self) -> None:
        from kiss.core.models.model_info import model

        m = model("text-embedding-004")
        assert m.model_name == "text-embedding-004"

    def test_model_minimax(self) -> None:
        from kiss.core.models.model_info import model

        m = model("minimax-m1")
        assert m.model_name == "minimax-m1"


class TestGetAvailableModels:
    def test_get_most_expensive_model(self) -> None:
        from kiss.core.models.model_info import get_most_expensive_model

        result = get_most_expensive_model()
        assert isinstance(result, str)

    def test_get_default_model_priority(self) -> None:
        """Test that get_default_model picks the right model per API key priority."""
        import os

        from kiss.core import config as config_module
        from kiss.core.models.model_info import get_default_model

        env_keys = [
            "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "TOGETHER_API_KEY",
        ]
        saved = {k: os.environ.get(k) for k in env_keys}
        try:
            for k in env_keys:
                os.environ.pop(k, None)
            config_module.DEFAULT_CONFIG = config_module.Config()
            import shutil

            if shutil.which("claude") is not None:
                assert get_default_model() == "cc/opus"
            else:
                assert get_default_model() == "claude-opus-4-6"

            os.environ["TOGETHER_API_KEY"] = "t"
            config_module.DEFAULT_CONFIG = config_module.Config()
            assert get_default_model() == "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"

            os.environ["OPENROUTER_API_KEY"] = "t"
            config_module.DEFAULT_CONFIG = config_module.Config()
            assert get_default_model() == "openrouter/anthropic/claude-opus-4.7"

            os.environ["GEMINI_API_KEY"] = "t"
            config_module.DEFAULT_CONFIG = config_module.Config()
            assert get_default_model() == "gemini-3.1-pro-preview"

            os.environ["OPENAI_API_KEY"] = "t"
            config_module.DEFAULT_CONFIG = config_module.Config()
            assert get_default_model() == "gpt-5.5"

            os.environ["ANTHROPIC_API_KEY"] = "t"
            config_module.DEFAULT_CONFIG = config_module.Config()
            assert get_default_model() == "claude-opus-4-7"
        finally:
            for k in env_keys:
                val = saved[k]
                if val is not None:
                    os.environ[k] = val
                else:
                    os.environ.pop(k, None)
            config_module.DEFAULT_CONFIG = config_module.Config()


class TestAttachment:
    def test_from_file_unsupported_mime(self, tmp_path: Path) -> None:
        from kiss.core.models.model import Attachment

        f = tmp_path / "test.xyz"
        f.write_bytes(b"data")
        with pytest.raises(ValueError, match="Unsupported MIME type"):
            Attachment.from_file(str(f))

    def test_from_file_not_found(self) -> None:
        from kiss.core.models.model import Attachment

        with pytest.raises(FileNotFoundError):
            Attachment.from_file("/nonexistent/file.png")


class TestRelentlessAgentDockerBash:
    def test_docker_bash_raises_without_manager(self) -> None:
        from kiss.core.kiss_error import KISSError
        from kiss.core.relentless_agent import RelentlessAgent

        agent = RelentlessAgent("test")
        agent._reset(
            model_name="gemini-3-flash-preview",
            max_sub_sessions=1,
            max_steps=3,
            max_budget=0.01,
            work_dir=None,
            docker_image=None,
        )
        with pytest.raises(KISSError, match="Docker manager not initialized"):
            agent._docker_bash("echo hi", "test")


class TestConfigBuilder:
    def test_add_config_twice_preserves_first(self) -> None:
        """Calling add_config twice preserves previous config fields."""
        from pydantic import BaseModel as PydanticBaseModel

        from kiss.core import config as config_module
        from kiss.core.config_builder import add_config

        original = config_module.DEFAULT_CONFIG

        class Cfg1(PydanticBaseModel):
            a: int = 1

        class Cfg2(PydanticBaseModel):
            b: int = 2

        try:
            add_config("cfg1", Cfg1)
            add_config("cfg2", Cfg2)
            cfg = config_module.DEFAULT_CONFIG
            assert cfg.cfg1.a == 1  # type: ignore[attr-defined]
            assert cfg.cfg2.b == 2  # type: ignore[attr-defined]
        finally:
            config_module.DEFAULT_CONFIG = original


class TestPrintToConsole:
    def test_format_result_summary_no_success(self) -> None:
        """Dict with summary but no success key should skip the success label."""
        import yaml

        from kiss.core.print_to_console import ConsolePrinter

        p = ConsolePrinter()
        content = yaml.dump({"summary": "Done without status"})
        result = p.print(content, type="result", total_tokens=0, cost=0.0)
        assert isinstance(result, str)


class TestAnthropicBuildKwargs:

    def test_build_kwargs_user_set_max_tokens_with_thinking(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-sonnet-4-test", api_key="test")
        m.model_config = {"max_tokens": 999}
        kwargs = m._build_create_kwargs()
        assert kwargs["max_tokens"] == 999
        assert "thinking" in kwargs

    def test_build_kwargs_custom_thinking_not_overridden(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-sonnet-4-test", api_key="test")
        m.model_config = {"thinking": {"type": "disabled"}}
        kwargs = m._build_create_kwargs()
        assert kwargs["thinking"] == {"type": "disabled"}


class TestAnthropicTokenCounts:
    def test_no_usage(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")

        class FakeResp:
            usage = None

        assert m.extract_input_output_token_counts_from_response(FakeResp()) == (0, 0, 0, 0)


class TestAnthropicHelpers:
    def test_normalize_list_of_objects(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")

        class TextBlock:
            type = "text"
            text = "hello"

        class ThinkBlock:
            type = "thinking"
            thinking = "hmm"

        result = m._normalize_content_blocks([TextBlock(), ThinkBlock()])
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "hello"}
        assert result[1] == {"type": "thinking", "thinking": "hmm"}

    def test_normalize_model_dump_block(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")

        class DumpBlock:
            type = "custom"

            def model_dump(self, exclude_none: bool = False) -> dict:
                return {"type": "custom", "data": "val"}

        result = m._normalize_content_blocks([DumpBlock()])
        assert result[0] == {"type": "custom", "data": "val"}

    def test_normalize_unknown_block_fallback(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")

        class WeirdBlock:
            pass

        result = m._normalize_content_blocks([WeirdBlock()])
        assert result[0]["type"] == "text"
        assert "WeirdBlock" in result[0]["text"]

    def test_normalize_tool_use_block(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")

        class ToolBlock:
            type = "tool_use"
            id = "tid"
            name = "fn"
            input = {"x": 1}

        result = m._normalize_content_blocks([ToolBlock()])
        assert result[0]["type"] == "tool_use"
        assert result[0]["name"] == "fn"


class TestAnthropicAddFunctionResults:

    def test_add_results_with_usage_info(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")
        m.set_usage_info_for_messages("Tokens: 100")
        m.conversation = [{"role": "assistant", "content": "text"}]
        m.add_function_results_to_conversation_and_return(
            [("fn1", {"result": "done"})]
        )
        last = m.conversation[-1]
        assert "Tokens: 100" in last["content"][0]["content"]



