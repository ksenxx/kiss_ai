"""Tests for refactoring tasks: ChannelConfig, _find_tool_call_ids,
_build_openai_tools_schema, _resolve_openai_tools_schema,
_build_text_based_tools_prompt, _parse_text_based_tool_calls,
_ArtifactDirProxy, and related helpers.

No mocks, patches, fakes, or any form of test doubles.
"""

from __future__ import annotations

from pathlib import Path

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

    def test_no_assistant_message(self) -> None:
        m = self._make_openai_model()
        m.conversation = [{"role": "user", "content": "hi"}]
        assert m._find_tool_call_ids_from_last_assistant() == []


class TestAddFunctionResults:
    """Tests for Model.add_function_results_to_conversation_and_return (OpenAI base)."""

    def _make_model(self) -> Model:
        m = model("gpt-4.1-mini")
        m.conversation = []
        m.usage_info_for_messages = ""
        return m

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


class TestAnthropicAddFunctionResults:
    """Tests for AnthropicModel.add_function_results_to_conversation_and_return."""

    def _make_anthropic_model(self) -> AnthropicModel:
        m = model("claude-haiku-4-5")
        assert isinstance(m, AnthropicModel)
        m.conversation = []
        m.usage_info_for_messages = ""
        return m

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


class TestParseTextBasedToolCalls:
    """Tests for _parse_text_based_tool_calls."""

    def test_tool_calls_not_list(self) -> None:
        content = '{"tool_calls": "not a list"}'
        calls = _parse_text_based_tool_calls(content)
        assert calls == []

    def test_tool_call_without_name(self) -> None:
        content = '{"tool_calls": [{"arguments": {"x": 1}}]}'
        calls = _parse_text_based_tool_calls(content)
        assert calls == []


# =========================================================================
# Task 8: _build_openai_tools_schema and _resolve_openai_tools_schema
# =========================================================================


class TestBuildOpenaiToolsSchema:
    """Tests for Model._build_openai_tools_schema."""

    def _make_model(self) -> Model:
        return model("gpt-4.1-mini")

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

    def test_unannotated_params_default_to_string(self) -> None:
        def untyped(x):
            """Untyped function."""
            return x

        m = self._make_model()
        schema = m._build_openai_tools_schema({"untyped": untyped})
        props = schema[0]["function"]["parameters"]["properties"]
        assert props["x"]["type"] == "string"


# =========================================================================
# Task 14: _ArtifactDirProxy
# =========================================================================


class TestArtifactDirProxy:
    """Tests for _ArtifactDirProxy lazy directory creation and thread-safety."""

    def test_proxy_hash(self) -> None:
        proxy = _ArtifactDirProxy()
        assert hash(proxy) == hash(str(proxy))

    def test_set_artifact_base_dir(self, tmp_path: Path) -> None:
        original = get_artifact_dir()
        try:
            set_artifact_base_dir(str(tmp_path))
            new_dir = get_artifact_dir()
            assert str(tmp_path) in new_dir
        finally:
            # Reset state - set back to a temp dir so other tests work
            set_artifact_base_dir(str(Path(original).parent))
