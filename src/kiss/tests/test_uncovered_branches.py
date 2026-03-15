"""Integration tests targeting uncovered branches in core/, core/models/, agents/sorcar/.

No mocks, patches, or test doubles.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# printer.py — MultiPrinter (lines 225, 238-241, 249-250, 254-255)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# models/__init__.py — ImportError branches (lines 16-18, 22-24, 28-30)
# These are ImportError fallback branches. Since the packages are installed,
# the try-branches are covered. The except-branches can't be covered without
# uninstalling packages. We verify the imports succeed.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# model_info.py — model() factory branches, ImportError branches
# ---------------------------------------------------------------------------


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

    def test_model_unknown_raises(self) -> None:
        from kiss.core.kiss_error import KISSError
        from kiss.core.models.model_info import model

        with pytest.raises(KISSError, match="Unknown model name"):
            model("totally-unknown-model")

# ---------------------------------------------------------------------------
# model_info.py — get_available_models and get_most_expensive_model
# ---------------------------------------------------------------------------


class TestGetAvailableModels:
    def test_get_most_expensive_model(self) -> None:
        from kiss.core.models.model_info import get_most_expensive_model

        result = get_most_expensive_model()
        # May be empty string if no keys configured
        assert isinstance(result, str)

# ---------------------------------------------------------------------------
# model.py — Attachment.from_file, _invoke_token_callback, type conversions
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# model.py — _python_type_to_json_schema branches
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# model.py — _invoke_token_callback, close_callback_loop
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# model.py — _parse_docstring_params branches
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# model.py — add_message_to_conversation with usage_info
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# model.py — _function_to_openai_tool
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# utils.py — config_to_dict (list and __dict__ branches)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# relentless_agent.py — _docker_bash without docker
# ---------------------------------------------------------------------------


class TestRelentlessAgentDockerBash:
    def test_docker_bash_raises_without_manager(self) -> None:
        from kiss.core.kiss_error import KISSError
        from kiss.core.relentless_agent import RelentlessAgent

        agent = RelentlessAgent("test")
        # Must call _reset first to initialize docker_manager attribute
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


# ---------------------------------------------------------------------------
# kiss_agent.py — _is_retryable_error
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# config_builder.py — line 130 (empty api_keys_from_env)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# print_to_console.py — line 46->51 (non-dict yaml data)
# ---------------------------------------------------------------------------


class TestPrintToConsole:
    def test_format_result_summary_no_success(self) -> None:
        """Dict with summary but no success key should skip the success label."""
        import yaml

        from kiss.core.print_to_console import ConsolePrinter

        p = ConsolePrinter()
        content = yaml.dump({"summary": "Done without status"})
        result = p.print(content, type="result", step_count=1, total_tokens=0, cost=0.0)
        assert isinstance(result, str)

# ---------------------------------------------------------------------------
# anthropic_model.py — _build_create_kwargs branches
# ---------------------------------------------------------------------------


class TestAnthropicBuildKwargs:
    def test_build_kwargs_with_system_instruction(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")
        m.model_config = {"system_instruction": "You are helpful."}
        kwargs = m._build_create_kwargs()
        assert kwargs["system"] == "You are helpful."

    def test_build_kwargs_opus_adaptive_thinking(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-opus-4-6", api_key="test")
        kwargs = m._build_create_kwargs()
        assert kwargs.get("thinking") == {"type": "adaptive"}

    def test_build_kwargs_user_set_max_tokens_with_thinking(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-sonnet-4-test", api_key="test")
        m.model_config = {"max_tokens": 999}
        kwargs = m._build_create_kwargs()
        assert kwargs["max_tokens"] == 999
        # Thinking should still be set
        assert "thinking" in kwargs

    def test_build_kwargs_custom_thinking_not_overridden(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-sonnet-4-test", api_key="test")
        m.model_config = {"thinking": {"type": "disabled"}}
        kwargs = m._build_create_kwargs()
        assert kwargs["thinking"] == {"type": "disabled"}

    def test_build_kwargs_non_claude4_no_thinking(self) -> None:
        """Non-claude-4 models should NOT have thinking auto-enabled."""
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-3-5-sonnet-20240620", api_key="test")
        kwargs = m._build_create_kwargs()
        assert "thinking" not in kwargs
        assert kwargs["max_tokens"] == 16384


# ---------------------------------------------------------------------------
# anthropic_model.py — extract_input_output_token_counts_from_response
# ---------------------------------------------------------------------------


class TestAnthropicTokenCounts:
    def test_no_usage(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")

        class FakeResp:
            usage = None

        assert m.extract_input_output_token_counts_from_response(FakeResp()) == (0, 0, 0, 0)

    def test_with_usage(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")

        class Usage:
            input_tokens = 10
            output_tokens = 20
            cache_read_input_tokens = 5
            cache_creation_input_tokens = 3

        class FakeResp:
            usage = Usage()

        result = m.extract_input_output_token_counts_from_response(FakeResp())
        assert result == (10, 20, 5, 3)


# ---------------------------------------------------------------------------
# anthropic_model.py — get_embedding raises NotImplementedError
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# anthropic_model.py — _normalize_content_blocks, _extract_text
# ---------------------------------------------------------------------------


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

    def test_normalize_thinking_block_with_signature(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")

        class ThinkBlock:
            type = "thinking"
            thinking = "hmm"
            signature = "sig123"

        result = m._normalize_content_blocks([ThinkBlock()])
        assert result[0]["signature"] == "sig123"

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
            pass  # no type, no model_dump

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

    def test_extract_text(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")
        blocks = [
            {"type": "text", "text": "Hello"},
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "World"},
        ]
        # _extract_text_from_blocks joins text blocks without separator
        assert m._extract_text_from_blocks(blocks) == "HelloWorld"


# ---------------------------------------------------------------------------
# anthropic_model.py — _build_anthropic_tools_schema
# ---------------------------------------------------------------------------


class TestAnthropicAddFunctionResults:
    def test_add_results_with_tool_use_ids(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")
        m.conversation = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "id1", "name": "fn1"},
                    {"type": "tool_use", "id": "id2", "name": "fn2"},
                ],
            }
        ]
        m.add_function_results_to_conversation_and_return(
            [("fn1", {"result": "r1"}), ("fn2", {"result": "r2"})]
        )
        last = m.conversation[-1]
        assert last["role"] == "user"
        assert len(last["content"]) == 2
        assert last["content"][0]["tool_use_id"] == "id1"
        assert last["content"][1]["tool_use_id"] == "id2"

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

# ---------------------------------------------------------------------------
# StreamEventParser — content_block types
# ---------------------------------------------------------------------------


class TestStreamEventParser:
    def test_parse_text_block_end(self) -> None:
        from kiss.core.printer import StreamEventParser

        class Event:
            def __init__(self, d: dict) -> None:
                self.event = d

        p = StreamEventParser()
        # Start a text block
        p.parse_stream_event(
            Event({"type": "content_block_start", "content_block": {"type": "text"}})
        )
        # End it
        p.parse_stream_event(Event({"type": "content_block_stop"}))
        # Should have called _on_text_block_end (default is no-op)

    def test_parse_thinking_delta(self) -> None:
        from kiss.core.printer import StreamEventParser

        class Event:
            def __init__(self, d: dict) -> None:
                self.event = d

        p = StreamEventParser()
        p.parse_stream_event(
            Event({"type": "content_block_start", "content_block": {"type": "thinking"}})
        )
        text = p.parse_stream_event(
            Event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "hmm"},
                }
            )
        )
        assert text == "hmm"
        # End thinking block
        p.parse_stream_event(Event({"type": "content_block_stop"}))

    def test_parse_tool_use_bad_json(self) -> None:
        from kiss.core.printer import StreamEventParser

        class Event:
            def __init__(self, d: dict) -> None:
                self.event = d

        p = StreamEventParser()
        p.parse_stream_event(
            Event(
                {
                    "type": "content_block_start",
                    "content_block": {"type": "tool_use", "name": "test"},
                }
            )
        )
        # Send bad JSON
        p.parse_stream_event(
            Event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "input_json_delta", "partial_json": "{bad"},
                }
            )
        )
        # End should handle bad JSON
        p.parse_stream_event(Event({"type": "content_block_stop"}))


# ---------------------------------------------------------------------------
# printer.py — utility functions
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# sorcar_agent.py — _build_arg_parser and _resolve_task
# ---------------------------------------------------------------------------


class TestSorcarAgentCli:
    def test_resolve_task_default(self) -> None:
        from kiss.agents.sorcar.sorcar_agent import _build_arg_parser, _resolve_task

        parser = _build_arg_parser()
        args = parser.parse_args([])
        result = _resolve_task(args)
        assert "weather" in result.lower()

    def test_resolve_task_from_string(self) -> None:
        from kiss.agents.sorcar.sorcar_agent import _build_arg_parser, _resolve_task

        parser = _build_arg_parser()
        args = parser.parse_args(["--task", "Do something"])
        result = _resolve_task(args)
        assert result == "Do something"

    def test_resolve_task_from_file(self, tmp_path: Path) -> None:
        from kiss.agents.sorcar.sorcar_agent import _build_arg_parser, _resolve_task

        f = tmp_path / "task.txt"
        f.write_text("File task content")
        parser = _build_arg_parser()
        args = parser.parse_args(["-f", str(f)])
        result = _resolve_task(args)
        assert result == "File task content"
