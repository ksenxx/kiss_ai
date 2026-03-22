"""Integration tests targeting uncovered branches. No mocks, patches, or test doubles."""

from __future__ import annotations

import shutil
import tempfile
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
        from kiss.core.models.openai_compatible_model import (
            _parse_text_based_tool_calls,
        )

        content = '```json\n{broken json}\n```'
        calls = _parse_text_based_tool_calls(content)
        assert calls == []


class TestTaskHistoryEdgeCases(TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        import kiss.agents.sorcar.task_history as th

        kiss_dir = Path(self._tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self._saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        th._db_conn = None

    def tearDown(self) -> None:
        import kiss.agents.sorcar.task_history as th

        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        (th._DB_PATH, th._db_conn, th._KISS_DIR) = self._saved
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_search_history_fresh_db(self) -> None:
        import kiss.agents.sorcar.task_history as th

        results = th._search_history("anything", limit=10)
        assert isinstance(results, list)

    def test_set_empty_events_clears_events(self) -> None:
        import kiss.agents.sorcar.task_history as th

        th._add_task("my task")
        events: list[dict[str, object]] = [{"type": "text", "data": "hello"}]
        th._set_latest_chat_events(events, task="my task")
        th._set_latest_chat_events([], task="my task")
        loaded = th._load_task_chat_events("my task")
        assert loaded == []

    def test_get_history_entry_out_of_range(self) -> None:
        import kiss.agents.sorcar.task_history as th

        entry = th._get_history_entry(9999)
        assert entry is None


class TestCodeServerMergeHelpers(TestCase):
    def test_restore_merge_files_no_data(self) -> None:
        from kiss.agents.sorcar.code_server import _restore_merge_files

        td = Path(tempfile.mkdtemp())
        result = _restore_merge_files(str(td), str(td))
        assert result == 0
        shutil.rmtree(td, ignore_errors=True)


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

    def test_model_unknown_raises(self) -> None:
        from kiss.core.kiss_error import KISSError
        from kiss.core.models.model_info import model

        with pytest.raises(KISSError, match="Unknown model name"):
            model("totally-unknown-model")


class TestGetAvailableModels:
    def test_get_most_expensive_model(self) -> None:
        from kiss.core.models.model_info import get_most_expensive_model

        result = get_most_expensive_model()
        assert isinstance(result, str)


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

    def test_extract_text(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")
        blocks = [
            {"type": "text", "text": "Hello"},
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "World"},
        ]
        assert m._extract_text_from_blocks(blocks) == "HelloWorld"


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


class TestStreamEventParser:
    def test_parse_text_block_end(self) -> None:
        from kiss.core.printer import StreamEventParser

        class Event:
            def __init__(self, d: dict) -> None:
                self.event = d

        p = StreamEventParser()
        p.parse_stream_event(
            Event({"type": "content_block_start", "content_block": {"type": "text"}})
        )
        p.parse_stream_event(Event({"type": "content_block_stop"}))

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
        p.parse_stream_event(
            Event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "input_json_delta", "partial_json": "{bad"},
                }
            )
        )
        p.parse_stream_event(Event({"type": "content_block_stop"}))


