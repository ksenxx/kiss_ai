"""Tests for refactoring tasks: ChannelConfig, _find_tool_call_ids,
_build_openai_tools_schema, _resolve_openai_tools_schema,
_build_text_based_tools_prompt, _parse_text_based_tool_calls,
_ArtifactDirProxy, and related helpers.

No mocks, patches, fakes, or any form of test doubles.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

from kiss.channels._channel_agent_utils import ChannelConfig
from kiss.core.config import (
    _ArtifactDirProxy,
    get_artifact_dir,
    set_artifact_base_dir,
)
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model import (
    Model,
    _build_text_based_tools_prompt,
    _parse_text_based_tool_calls,
)
from kiss.core.models.model_info import model

# =========================================================================
# Task 4: ChannelConfig
# =========================================================================


class TestChannelConfig:
    """Integration tests for ChannelConfig: save, load, clear, missing keys, permissions."""

    def test_save_load_clear(self, tmp_path: Path) -> None:
        cfg = ChannelConfig(tmp_path, ("token",))
        cfg.save({"token": "abc123", "extra": "val"})
        loaded = cfg.load()
        assert loaded == {"token": "abc123", "extra": "val"}
        cfg.clear()
        assert cfg.load() is None
        assert not cfg.path.exists()

    def test_missing_required_keys_returns_none(self, tmp_path: Path) -> None:
        cfg = ChannelConfig(tmp_path, ("token", "secret"))
        cfg.save({"token": "abc123"})
        assert cfg.load() is None

    def test_empty_required_key_returns_none(self, tmp_path: Path) -> None:
        cfg = ChannelConfig(tmp_path, ("token",))
        cfg.save({"token": ""})
        assert cfg.load() is None

    def test_nonexistent_file_returns_none(self, tmp_path: Path) -> None:
        cfg = ChannelConfig(tmp_path, ())
        assert cfg.load() is None

    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        cfg = ChannelConfig(tmp_path, ())
        cfg.path.parent.mkdir(parents=True, exist_ok=True)
        cfg.path.write_text("{bad json!!")
        assert cfg.load() is None

    def test_non_dict_json_returns_none(self, tmp_path: Path) -> None:
        cfg = ChannelConfig(tmp_path, ())
        cfg.path.parent.mkdir(parents=True, exist_ok=True)
        cfg.path.write_text('"just a string"')
        assert cfg.load() is None

    def test_no_required_keys(self, tmp_path: Path) -> None:
        cfg = ChannelConfig(tmp_path, ())
        cfg.save({"any_key": "val"})
        assert cfg.load() == {"any_key": "val"}

    def test_clear_nonexistent_file(self, tmp_path: Path) -> None:
        cfg = ChannelConfig(tmp_path, ())
        cfg.clear()  # Should not raise

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix permissions")
    def test_file_permissions(self, tmp_path: Path) -> None:
        cfg = ChannelConfig(tmp_path, ())
        cfg.save({"key": "val"})
        assert cfg.path.stat().st_mode & 0o777 == 0o600

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        cfg = ChannelConfig(nested, ())
        cfg.save({"key": "val"})
        assert cfg.path.exists()
        assert cfg.load() == {"key": "val"}

    def test_null_value_becomes_empty_string(self, tmp_path: Path) -> None:
        cfg = ChannelConfig(tmp_path, ())
        cfg.path.parent.mkdir(parents=True, exist_ok=True)
        cfg.path.write_text(json.dumps({"key": None}))
        loaded = cfg.load()
        assert loaded == {"key": ""}


# =========================================================================
# Task 3: _find_tool_call_ids and add_function_results_to_conversation_and_return
# =========================================================================


class TestFindToolCallIds:
    """Tests for Model._find_tool_call_ids_from_last_assistant."""

    def _make_openai_model(self) -> Model:
        """Create a minimal model with OpenAI-style conversation."""
        m = model("gpt-4.1-mini")
        m.conversation = []
        return m

    def test_openai_style_tool_calls(self) -> None:
        m = self._make_openai_model()
        m.conversation = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "foo"}, "id": "call_1"},
                    {"function": {"name": "bar"}, "id": "call_2"},
                ],
            },
        ]
        ids = m._find_tool_call_ids_from_last_assistant()
        assert ids == [("foo", "call_1"), ("bar", "call_2")]

    def test_anthropic_style_tool_use(self) -> None:
        m = self._make_openai_model()
        m.conversation = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll do that"},
                    {"type": "tool_use", "name": "search", "id": "toolu_abc"},
                ],
            },
        ]
        ids = m._find_tool_call_ids_from_last_assistant()
        assert ids == [("search", "toolu_abc")]

    def test_empty_conversation(self) -> None:
        m = self._make_openai_model()
        m.conversation = []
        assert m._find_tool_call_ids_from_last_assistant() == []

    def test_no_assistant_message(self) -> None:
        m = self._make_openai_model()
        m.conversation = [{"role": "user", "content": "hi"}]
        assert m._find_tool_call_ids_from_last_assistant() == []

    def test_assistant_no_tool_calls(self) -> None:
        m = self._make_openai_model()
        m.conversation = [
            {"role": "assistant", "content": "Hello there!"},
        ]
        assert m._find_tool_call_ids_from_last_assistant() == []


class TestAddFunctionResults:
    """Tests for Model.add_function_results_to_conversation_and_return (OpenAI base)."""

    def _make_model(self) -> Model:
        m = model("gpt-4.1-mini")
        m.conversation = []
        m.usage_info_for_messages = ""
        return m

    def test_multiple_results_matching_tool_calls(self) -> None:
        m = self._make_model()
        m.conversation = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "foo"}, "id": "call_1"},
                    {"function": {"name": "bar"}, "id": "call_2"},
                ],
            },
        ]
        m.add_function_results_to_conversation_and_return(
            [
                ("foo", {"result": "result_a"}),
                ("bar", {"result": "result_b"}),
            ]
        )
        assert len(m.conversation) == 3  # assistant + 2 tool messages
        assert m.conversation[1]["role"] == "tool"
        assert m.conversation[1]["tool_call_id"] == "call_1"
        assert m.conversation[1]["content"] == "result_a"
        assert m.conversation[2]["tool_call_id"] == "call_2"
        assert m.conversation[2]["content"] == "result_b"

    def test_fallback_id_when_count_mismatch(self) -> None:
        m = self._make_model()
        m.conversation = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "foo"}, "id": "call_1"},
                ],
            },
        ]
        m.add_function_results_to_conversation_and_return(
            [
                ("foo", {"result": "result_a"}),
                ("extra", {"result": "result_b"}),
            ]
        )
        assert m.conversation[2]["tool_call_id"] == "call_extra_1"

    def test_usage_info_appended(self) -> None:
        m = self._make_model()
        m.usage_info_for_messages = "[Budget: $0.50]"
        m.conversation = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "foo"}, "id": "call_1"},
                ],
            },
        ]
        m.add_function_results_to_conversation_and_return(
            [
                ("foo", {"result": "done"}),
            ]
        )
        assert "[Budget: $0.50]" in m.conversation[1]["content"]


class TestAnthropicAddFunctionResults:
    """Tests for AnthropicModel.add_function_results_to_conversation_and_return."""

    def _make_anthropic_model(self) -> AnthropicModel:
        m = model("claude-haiku-4-5")
        assert isinstance(m, AnthropicModel)
        m.conversation = []
        m.usage_info_for_messages = ""
        return m

    def test_with_explicit_tool_use_id(self) -> None:
        m = self._make_anthropic_model()
        m.conversation = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "foo", "id": "toolu_xyz"},
                ],
            },
        ]
        m.add_function_results_to_conversation_and_return(
            [
                ("foo", {"result": "ok", "tool_use_id": "toolu_explicit"}),
            ]
        )
        assert len(m.conversation) == 2
        result_msg = m.conversation[1]
        assert result_msg["role"] == "user"
        assert result_msg["content"][0]["tool_use_id"] == "toolu_explicit"

    def test_position_match_fallback(self) -> None:
        m = self._make_anthropic_model()
        m.conversation = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "foo", "id": "toolu_abc"},
                    {"type": "tool_use", "name": "bar", "id": "toolu_def"},
                ],
            },
        ]
        m.add_function_results_to_conversation_and_return(
            [
                ("foo", {"result": "a"}),
                ("bar", {"result": "b"}),
            ]
        )
        result_msg = m.conversation[1]
        assert result_msg["content"][0]["tool_use_id"] == "toolu_abc"
        assert result_msg["content"][1]["tool_use_id"] == "toolu_def"

    def test_fallback_id_generation(self) -> None:
        m = self._make_anthropic_model()
        m.conversation = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "foo", "id": "toolu_abc"},
                ],
            },
        ]
        m.add_function_results_to_conversation_and_return(
            [
                ("foo", {"result": "a"}),
                ("extra", {"result": "b"}),
            ]
        )
        result_msg = m.conversation[1]
        assert result_msg["content"][1]["tool_use_id"] == "toolu_extra_1"


# =========================================================================
# Task 7: _build_text_based_tools_prompt and _parse_text_based_tool_calls
# =========================================================================


class TestBuildTextBasedToolsPrompt:
    """Tests for _build_text_based_tools_prompt."""

    def test_empty_function_map_returns_empty(self) -> None:
        assert _build_text_based_tools_prompt({}) == ""

    def test_single_function_with_typed_params(self) -> None:
        def greet(name: str, times: int = 1) -> str:
            """Say hello to someone.

            Args:
                name: The person's name.
                times: How many times to greet.
            """
            return f"Hello {name}!" * times

        prompt = _build_text_based_tools_prompt({"greet": greet})
        assert "greet" in prompt
        assert "name" in prompt
        assert "times" in prompt
        assert "tool_calls" in prompt

    def test_function_without_annotations(self) -> None:
        def noop():
            """Do nothing."""
            pass

        prompt = _build_text_based_tools_prompt({"noop": noop})
        assert "noop" in prompt
        assert "no parameters" in prompt

    def test_multiple_functions(self) -> None:
        def foo(x: str) -> str:
            """Foo function."""
            return x

        def bar(y: int) -> int:
            """Bar function."""
            return y

        prompt = _build_text_based_tools_prompt({"foo": foo, "bar": bar})
        assert "foo" in prompt
        assert "bar" in prompt


class TestParseTextBasedToolCalls:
    """Tests for _parse_text_based_tool_calls."""

    def test_json_in_code_block(self) -> None:
        content = '```json\n{"tool_calls": [{"name": "search", "arguments": {"q": "hello"}}]}\n```'
        calls = _parse_text_based_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "search"
        assert calls[0]["arguments"] == {"q": "hello"}
        assert calls[0]["id"].startswith("call_")

    def test_clean_json(self) -> None:
        content = '{"tool_calls": [{"name": "finish", "arguments": {"result": "done"}}]}'
        calls = _parse_text_based_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "finish"

    def test_malformed_json_returns_empty(self) -> None:
        content = "This is just plain text with no JSON"
        calls = _parse_text_based_tool_calls(content)
        assert calls == []

    def test_json_without_tool_calls_key(self) -> None:
        content = '{"result": "hello"}'
        calls = _parse_text_based_tool_calls(content)
        assert calls == []

    def test_tool_calls_not_list(self) -> None:
        content = '{"tool_calls": "not a list"}'
        calls = _parse_text_based_tool_calls(content)
        assert calls == []

    def test_tool_call_without_name(self) -> None:
        content = '{"tool_calls": [{"arguments": {"x": 1}}]}'
        calls = _parse_text_based_tool_calls(content)
        assert calls == []

    def test_multiple_tool_calls(self) -> None:
        content = (
            '{"tool_calls": [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"x": 1}}]}'
        )
        calls = _parse_text_based_tool_calls(content)
        assert len(calls) == 2
        assert calls[0]["name"] == "a"
        assert calls[1]["name"] == "b"

    def test_generic_code_block(self) -> None:
        content = '```\n{"tool_calls": [{"name": "search", "arguments": {}}]}\n```'
        calls = _parse_text_based_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "search"


# =========================================================================
# Task 8: _build_openai_tools_schema and _resolve_openai_tools_schema
# =========================================================================


class TestBuildOpenaiToolsSchema:
    """Tests for Model._build_openai_tools_schema."""

    def _make_model(self) -> Model:
        return model("gpt-4.1-mini")

    def test_basic_function(self) -> None:
        def hello(name: str) -> str:
            """Say hello."""
            return f"Hello {name}"

        m = self._make_model()
        schema = m._build_openai_tools_schema({"hello": hello})
        assert len(schema) == 1
        assert schema[0]["type"] == "function"
        assert schema[0]["function"]["name"] == "hello"
        props = schema[0]["function"]["parameters"]["properties"]
        assert "name" in props
        assert props["name"]["type"] == "string"
        assert schema[0]["function"]["parameters"]["required"] == ["name"]

    def test_function_with_optional_params(self) -> None:
        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone."""
            return f"{greeting} {name}"

        m = self._make_model()
        schema = m._build_openai_tools_schema({"greet": greet})
        required = schema[0]["function"]["parameters"]["required"]
        assert "name" in required
        assert "greeting" not in required

    def test_various_types(self) -> None:

        def multi(a, b, c, d, e):  # type: ignore[no-untyped-def]
            """Multi-type function."""
            return ""

        # Manually set annotations to real types (avoids PEP 649/563 issues)
        multi.__annotations__ = {
            "a": int,
            "b": float,
            "c": bool,
            "d": list[str],
            "e": dict[str, str],
            "return": str,
        }

        m = self._make_model()
        schema = m._build_openai_tools_schema({"multi": multi})
        props = schema[0]["function"]["parameters"]["properties"]
        assert props["a"]["type"] == "integer"
        assert props["b"]["type"] == "number"
        assert props["c"]["type"] == "boolean"
        assert props["d"]["type"] == "array"
        assert props["d"]["items"]["type"] == "string"
        assert props["e"]["type"] == "object"

    def test_empty_function_map(self) -> None:
        m = self._make_model()
        schema = m._build_openai_tools_schema({})
        assert schema == []

    def test_unannotated_params_default_to_string(self) -> None:
        def untyped(x):
            """Untyped function."""
            return x

        m = self._make_model()
        schema = m._build_openai_tools_schema({"untyped": untyped})
        props = schema[0]["function"]["parameters"]["properties"]
        assert props["x"]["type"] == "string"


class TestResolveOpenaiToolsSchema:
    """Tests for Model._resolve_openai_tools_schema."""

    def test_returns_prebuilt_when_provided(self) -> None:
        m = model("gpt-4.1-mini")
        prebuilt = [{"type": "function", "function": {"name": "test"}}]
        result = m._resolve_openai_tools_schema({"test": lambda: None}, prebuilt)
        assert result is prebuilt

    def test_builds_schema_when_none(self) -> None:
        def hello(name: str) -> str:
            """Say hello."""
            return f"Hello {name}"

        m = model("gpt-4.1-mini")
        result = m._resolve_openai_tools_schema({"hello": hello}, None)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "hello"


# =========================================================================
# Task 14: _ArtifactDirProxy
# =========================================================================


class TestArtifactDirProxy:
    """Tests for _ArtifactDirProxy lazy directory creation and thread-safety."""

    def test_proxy_creates_directory_lazily(self) -> None:
        proxy = _ArtifactDirProxy()
        path_str = str(proxy)
        assert Path(path_str).exists()

    def test_proxy_fspath(self) -> None:
        import os

        proxy = _ArtifactDirProxy()
        fspath = os.fspath(proxy)
        assert isinstance(fspath, str)
        assert Path(fspath).exists()

    def test_proxy_equality(self) -> None:
        proxy = _ArtifactDirProxy()
        assert proxy == str(proxy)

    def test_proxy_hash(self) -> None:
        proxy = _ArtifactDirProxy()
        assert hash(proxy) == hash(str(proxy))

    def test_thread_safety(self) -> None:
        """Multiple threads calling get_artifact_dir get the same path."""
        results: list[str] = []

        def get_dir() -> None:
            results.append(get_artifact_dir())

        threads = [threading.Thread(target=get_dir) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(results)) == 1

    def test_set_artifact_base_dir(self, tmp_path: Path) -> None:
        original = get_artifact_dir()
        try:
            set_artifact_base_dir(str(tmp_path))
            new_dir = get_artifact_dir()
            assert str(tmp_path) in new_dir
        finally:
            # Reset state - set back to a temp dir so other tests work
            set_artifact_base_dir(str(Path(original).parent))
