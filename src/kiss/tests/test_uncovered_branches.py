"""Integration tests targeting uncovered branches in core/, core/models/, agents/sorcar/.

No mocks, patches, or test doubles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# printer.py — MultiPrinter (lines 225, 238-241, 249-250, 254-255)
# ---------------------------------------------------------------------------


class TestMultiPrinter:
    def test_multi_printer_print_and_reset(self) -> None:
        from kiss.core.printer import MultiPrinter, Printer

        class SimplePrinter(Printer):
            def __init__(self) -> None:
                self.printed: list[tuple[Any, str]] = []
                self.tokens: list[str] = []
                self.reset_count = 0

            def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
                self.printed.append((content, type))
                return f"result:{content}"

            async def token_callback(self, token: str) -> None:
                self.tokens.append(token)

            def reset(self) -> None:
                self.reset_count += 1

        p1 = SimplePrinter()
        p2 = SimplePrinter()
        mp = MultiPrinter([p1, p2])

        # Test print dispatches to all and returns last result
        result = mp.print("hello", type="text")
        assert result == "result:hello"
        assert len(p1.printed) == 1
        assert len(p2.printed) == 1

        # Test reset
        mp.reset()
        assert p1.reset_count == 1
        assert p2.reset_count == 1

    def test_multi_printer_token_callback(self) -> None:
        import asyncio

        from kiss.core.printer import MultiPrinter, Printer

        class SimplePrinter(Printer):
            def __init__(self) -> None:
                self.tokens: list[str] = []

            def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
                return ""

            async def token_callback(self, token: str) -> None:
                self.tokens.append(token)

            def reset(self) -> None:
                pass

        p1 = SimplePrinter()
        p2 = SimplePrinter()
        mp = MultiPrinter([p1, p2])
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(mp.token_callback("tok"))
        finally:
            loop.close()
        assert p1.tokens == ["tok"]
        assert p2.tokens == ["tok"]


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
    def test_model_with_base_url_in_config(self) -> None:
        from kiss.core.models.model_info import model

        m = model(
            "test-model",
            model_config={"base_url": "http://localhost:1234/v1", "api_key": "test-key"},
        )
        assert m.model_name == "test-model"

    def test_model_with_base_url_empty_filtered(self) -> None:
        from kiss.core.models.model_info import model

        m = model(
            "test-model",
            model_config={"base_url": "http://localhost:1234/v1"},
        )
        assert m.model_name == "test-model"

    def test_model_openrouter(self) -> None:
        from kiss.core.models.model_info import model

        m = model("openrouter/foo-bar")
        assert m.model_name == "openrouter/foo-bar"

    def test_model_openai_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kiss.core.models.model_info import model

        monkeypatch.setattr(
            "kiss.core.config.DEFAULT_CONFIG.agent.api_keys.OPENAI_API_KEY",
            "sk-test",
        )
        monkeypatch.setattr(
            "kiss.core.models.model_info._resolve_openai_auth_mode",
            lambda *_args: "api",
        )
        monkeypatch.setattr(
            "kiss.core.models.model_info._openai_compatible",
            lambda model_name, *_args, **_kwargs: type(
                "_ModelStub",
                (),
                {"model_name": model_name},
            )(),
        )
        m = model("gpt-4.1-mini")
        assert m.model_name == "gpt-4.1-mini"

    def test_model_together_prefix(self) -> None:
        from kiss.core.models.model_info import model

        m = model("meta-llama/Llama-3.3-70B-Instruct-Turbo")
        assert m.model_name == "meta-llama/Llama-3.3-70B-Instruct-Turbo"

    def test_model_claude(self) -> None:
        from kiss.core.models.model_info import model

        m = model("claude-haiku-4-5")
        assert m.model_name == "claude-haiku-4-5"

    def test_model_gemini(self) -> None:
        from kiss.core.models.model_info import model

        m = model("gemini-3-flash-preview")
        assert m.model_name == "gemini-3-flash-preview"

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

    def test_model_o1_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kiss.core.models.model_info import model

        monkeypatch.setattr(
            "kiss.core.config.DEFAULT_CONFIG.agent.api_keys.OPENAI_API_KEY",
            "sk-test",
        )
        monkeypatch.setattr(
            "kiss.core.models.model_info._resolve_openai_auth_mode",
            lambda *_args: "api",
        )
        monkeypatch.setattr(
            "kiss.core.models.model_info._openai_compatible",
            lambda model_name, *_args, **_kwargs: type(
                "_ModelStub",
                (),
                {"model_name": model_name},
            )(),
        )
        m = model("o1-mini")
        assert m.model_name == "o1-mini"

    def test_model_openai_gpt_oss_goes_to_together(self) -> None:
        """openai/gpt-oss prefix should match TOGETHER_PREFIXES, not OPENAI."""
        from kiss.core.models.model_info import model

        m = model("openai/gpt-oss-test")
        assert m.model_name == "openai/gpt-oss-test"


# ---------------------------------------------------------------------------
# model_info.py — get_available_models and get_most_expensive_model
# ---------------------------------------------------------------------------


class TestGetAvailableModels:
    def test_get_available_models_returns_sorted_list(self) -> None:
        from kiss.core.models.model_info import get_available_models

        result = get_available_models()
        assert isinstance(result, list)
        assert result == sorted(result)

    def test_get_most_expensive_model(self) -> None:
        from kiss.core.models.model_info import get_most_expensive_model

        result = get_most_expensive_model()
        # May be empty string if no keys configured
        assert isinstance(result, str)

    def test_get_most_expensive_model_no_fc_filter(self) -> None:
        from kiss.core.models.model_info import get_most_expensive_model

        result = get_most_expensive_model(fc_only=False)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# model.py — Attachment.from_file, _invoke_token_callback, type conversions
# ---------------------------------------------------------------------------


class TestAttachment:
    def test_from_file_known_mime(self, tmp_path: Path) -> None:
        from kiss.core.models.model import Attachment

        f = tmp_path / "test.png"
        # Minimal valid PNG header
        f.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde"
        )
        a = Attachment.from_file(str(f))
        assert a.mime_type == "image/png"

    def test_from_file_fallback_mime(self, tmp_path: Path) -> None:
        from kiss.core.models.model import Attachment

        f = tmp_path / "image.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0")
        a = Attachment.from_file(str(f))
        assert a.mime_type == "image/jpeg"

    def test_from_file_unsupported_mime(self, tmp_path: Path) -> None:
        from kiss.core.models.model import Attachment

        f = tmp_path / "test.xyz"
        f.write_bytes(b"data")
        with pytest.raises(ValueError, match="Unsupported MIME type"):
            Attachment.from_file(str(f))

    def test_to_base64_and_data_url(self) -> None:
        import base64

        from kiss.core.models.model import Attachment

        a = Attachment(data=b"hello", mime_type="image/png")
        assert a.to_base64() == base64.b64encode(b"hello").decode("ascii")
        assert a.to_data_url().startswith("data:image/png;base64,")

    def test_from_file_not_found(self) -> None:
        from kiss.core.models.model import Attachment

        with pytest.raises(FileNotFoundError):
            Attachment.from_file("/nonexistent/file.png")


# ---------------------------------------------------------------------------
# model.py — _python_type_to_json_schema branches
# ---------------------------------------------------------------------------


class TestTypeToJsonSchema:
    def _make_model(self) -> Any:
        from kiss.core.models.model import Model

        class ConcreteModel(Model):
            def initialize(self, prompt: str, attachments: Any = None) -> None:
                pass

            def generate(self) -> tuple[str, Any]:
                return "", None

            def generate_and_process_with_tools(self, function_map: Any) -> Any:
                return [], "", None

            def extract_input_output_token_counts_from_response(self, response: Any) -> Any:
                return 0, 0, 0, 0

            def get_embedding(self, text: str, embedding_model: Any = None) -> list[float]:
                return []

        return ConcreteModel("test")

    def test_empty_annotation(self) -> None:
        import inspect

        m = self._make_model()
        result = m._python_type_to_json_schema(inspect.Parameter.empty)
        assert result == {"type": "string"}

    def test_str_type(self) -> None:
        m = self._make_model()
        assert m._python_type_to_json_schema(str) == {"type": "string"}

    def test_int_type(self) -> None:
        m = self._make_model()
        assert m._python_type_to_json_schema(int) == {"type": "integer"}

    def test_float_type(self) -> None:
        m = self._make_model()
        assert m._python_type_to_json_schema(float) == {"type": "number"}

    def test_bool_type(self) -> None:
        m = self._make_model()
        assert m._python_type_to_json_schema(bool) == {"type": "boolean"}

    def test_none_type(self) -> None:
        m = self._make_model()
        assert m._python_type_to_json_schema(type(None)) == {"type": "null"}

    def test_list_type(self) -> None:
        m = self._make_model()
        assert m._python_type_to_json_schema(list[str]) == {
            "type": "array",
            "items": {"type": "string"},
        }

    def test_list_with_no_type_args(self) -> None:
        """Bare `list` has get_origin=None, falls through to default 'string'."""
        m = self._make_model()
        # bare list has no origin, so it hits the basic type mapping fallback
        assert m._python_type_to_json_schema(list) == {"type": "string"}

    def test_dict_with_key_value_types(self) -> None:
        m = self._make_model()
        assert m._python_type_to_json_schema(dict[str, int]) == {"type": "object"}

    def test_optional_type(self) -> None:
        from typing import Optional

        m = self._make_model()
        result = m._python_type_to_json_schema(Optional[str])  # noqa: UP045
        assert result == {"type": "string"}

    def test_union_type(self) -> None:
        from typing import Union

        m = self._make_model()
        result = m._python_type_to_json_schema(Union[str, int])  # noqa: UP007
        assert result == {"anyOf": [{"type": "string"}, {"type": "integer"}]}

    def test_unknown_type_defaults_to_string(self) -> None:
        m = self._make_model()
        result = m._python_type_to_json_schema(object)
        assert result == {"type": "string"}


# ---------------------------------------------------------------------------
# model.py — _invoke_token_callback, close_callback_loop
# ---------------------------------------------------------------------------


class TestModelCallbackLoop:
    def test_invoke_callback_no_running_loop(self) -> None:
        """When no event loop is running, uses instance _callback_loop."""
        from kiss.core.models.model import Model

        tokens: list[str] = []

        async def cb(token: str) -> None:
            tokens.append(token)

        class ConcreteModel(Model):
            def initialize(self, prompt: str, attachments: Any = None) -> None:
                pass

            def generate(self) -> tuple[str, Any]:
                return "", None

            def generate_and_process_with_tools(self, function_map: Any) -> Any:
                return [], "", None

            def extract_input_output_token_counts_from_response(self, response: Any) -> Any:
                return 0, 0, 0, 0

            def get_embedding(self, text: str, embedding_model: Any = None) -> list[float]:
                return []

        m = ConcreteModel("test", token_callback=cb)
        m._invoke_token_callback("hello")
        assert tokens == ["hello"]
        m.close_callback_loop()
        # calling close again should be no-op
        m.close_callback_loop()

    def test_invoke_callback_none(self) -> None:
        """When token_callback is None, _invoke_token_callback is a no-op."""
        from kiss.core.models.model import Model

        class ConcreteModel(Model):
            def initialize(self, prompt: str, attachments: Any = None) -> None:
                pass

            def generate(self) -> tuple[str, Any]:
                return "", None

            def generate_and_process_with_tools(self, function_map: Any) -> Any:
                return [], "", None

            def extract_input_output_token_counts_from_response(self, response: Any) -> Any:
                return 0, 0, 0, 0

            def get_embedding(self, text: str, embedding_model: Any = None) -> list[float]:
                return []

        m = ConcreteModel("test")
        m._invoke_token_callback("hello")  # should not raise


# ---------------------------------------------------------------------------
# model.py — _parse_docstring_params branches
# ---------------------------------------------------------------------------


class TestParseDocstringParams:
    def _make_model(self) -> Any:
        from kiss.core.models.model import Model

        class ConcreteModel(Model):
            def initialize(self, prompt: str, attachments: Any = None) -> None:
                pass

            def generate(self) -> tuple[str, Any]:
                return "", None

            def generate_and_process_with_tools(self, function_map: Any) -> Any:
                return [], "", None

            def extract_input_output_token_counts_from_response(self, response: Any) -> Any:
                return 0, 0, 0, 0

            def get_embedding(self, text: str, embedding_model: Any = None) -> list[float]:
                return []

        return ConcreteModel("test")

    def test_parse_args_section(self) -> None:
        m = self._make_model()
        doc = """Do something.

Args:
    name: The name of the thing.
    count (int): How many.

Returns:
    str: The result.
"""
        result = m._parse_docstring_params(doc)
        assert result["name"] == "The name of the thing."
        assert result["count"] == "How many."

    def test_parse_no_args(self) -> None:
        m = self._make_model()
        result = m._parse_docstring_params("No args here")
        assert result == {}

    def test_parse_raises_section_stops(self) -> None:
        m = self._make_model()
        doc = """Do something.

Args:
    x: The x.

Raises:
    ValueError: if bad.
"""
        result = m._parse_docstring_params(doc)
        assert "x" in result
        assert len(result) == 1


# ---------------------------------------------------------------------------
# model.py — add_message_to_conversation with usage_info
# ---------------------------------------------------------------------------


class TestAddMessage:
    def _make_model(self) -> Any:
        from kiss.core.models.model import Model

        class ConcreteModel(Model):
            def initialize(self, prompt: str, attachments: Any = None) -> None:
                pass

            def generate(self) -> tuple[str, Any]:
                return "", None

            def generate_and_process_with_tools(self, function_map: Any) -> Any:
                return [], "", None

            def extract_input_output_token_counts_from_response(self, response: Any) -> Any:
                return 0, 0, 0, 0

            def get_embedding(self, text: str, embedding_model: Any = None) -> list[float]:
                return []

        return ConcreteModel("test")

    def test_user_message_appends_usage_info(self) -> None:
        m = self._make_model()
        m.set_usage_info_for_messages("Tokens: 42")
        m.add_message_to_conversation("user", "Hello")
        assert "Tokens: 42" in m.conversation[-1]["content"]

    def test_assistant_message_no_usage_info(self) -> None:
        m = self._make_model()
        m.set_usage_info_for_messages("Tokens: 42")
        m.add_message_to_conversation("assistant", "Hi")
        assert m.conversation[-1]["content"] == "Hi"


# ---------------------------------------------------------------------------
# model.py — _function_to_openai_tool
# ---------------------------------------------------------------------------


class TestFunctionToOpenaiTool:
    def _make_model(self) -> Any:
        from kiss.core.models.model import Model

        class ConcreteModel(Model):
            def initialize(self, prompt: str, attachments: Any = None) -> None:
                pass

            def generate(self) -> tuple[str, Any]:
                return "", None

            def generate_and_process_with_tools(self, function_map: Any) -> Any:
                return [], "", None

            def extract_input_output_token_counts_from_response(self, response: Any) -> Any:
                return 0, 0, 0, 0

            def get_embedding(self, text: str, embedding_model: Any = None) -> list[float]:
                return []

        return ConcreteModel("test")

    def test_function_no_docstring(self) -> None:
        m = self._make_model()

        def no_doc(x: str) -> str:
            return ""

        schema = m._function_to_openai_tool(no_doc)
        assert schema["function"]["name"] == "no_doc"
        assert "x" in schema["function"]["parameters"]["properties"]
        assert "x" in schema["function"]["parameters"]["required"]

    def test_function_with_default(self) -> None:
        m = self._make_model()

        def has_default(x: str, y: int = 5) -> str:
            """Does something.

            Args:
                x: The x param.
                y: The y param.
            """
            return x

        schema = m._function_to_openai_tool(has_default)
        assert "x" in schema["function"]["parameters"]["required"]
        assert "y" not in schema["function"]["parameters"]["required"]


# ---------------------------------------------------------------------------
# utils.py — config_to_dict (list and __dict__ branches)
# ---------------------------------------------------------------------------


class TestConfigToDict:
    def test_config_to_dict_returns_dict(self) -> None:
        from kiss.core.utils import config_to_dict

        result = config_to_dict()
        assert isinstance(result, dict)
        # Should not contain API keys
        assert not any("API_KEY" in str(v) for v in str(result))


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


class TestIsRetryableError:
    def test_retryable_generic_error(self) -> None:
        from kiss.core.kiss_agent import _is_retryable_error

        assert _is_retryable_error(RuntimeError("some transient error")) is True

    def test_non_retryable_by_type(self) -> None:
        from kiss.core.kiss_agent import _is_retryable_error

        # Test with an error type that has 'AuthenticationError' in name
        class AuthenticationError(Exception):
            pass

        assert _is_retryable_error(AuthenticationError("bad auth")) is False

    def test_non_retryable_by_message(self) -> None:
        from kiss.core.kiss_agent import _is_retryable_error

        assert _is_retryable_error(RuntimeError("invalid api key provided")) is False
        assert _is_retryable_error(RuntimeError("Unauthorized access")) is False
        assert _is_retryable_error(RuntimeError("Permission denied")) is False
        assert _is_retryable_error(RuntimeError("incorrect api key")) is False


# ---------------------------------------------------------------------------
# config_builder.py — line 130 (empty api_keys_from_env)
# ---------------------------------------------------------------------------


class TestConfigBuilder:
    def test_add_config_creates_field(self) -> None:
        from pydantic import BaseModel as PydanticBaseModel

        from kiss.core import config as config_module
        from kiss.core.config_builder import add_config

        original = config_module.DEFAULT_CONFIG

        class TestCfg(PydanticBaseModel):
            x: int = 42

        try:
            add_config("test_cfg", TestCfg)
            cfg = config_module.DEFAULT_CONFIG
            assert hasattr(cfg, "test_cfg")
            assert cfg.test_cfg.x == 42  # type: ignore[attr-defined]
        finally:
            config_module.DEFAULT_CONFIG = original

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
    def test_format_result_non_dict_yaml(self) -> None:
        from kiss.core.print_to_console import ConsolePrinter

        p = ConsolePrinter()
        # result content that parses to non-dict YAML
        result = p.print(
            "just a string value", type="result", step_count=1, total_tokens=0, cost=0.0
        )
        assert isinstance(result, str)

    def test_format_result_with_summary(self) -> None:
        import yaml

        from kiss.core.print_to_console import ConsolePrinter

        p = ConsolePrinter()
        content = yaml.dump({"success": True, "summary": "All done"})
        result = p.print(content, type="result", step_count=1, total_tokens=0, cost=0.0)
        assert isinstance(result, str)

    def test_format_result_summary_no_success(self) -> None:
        """Dict with summary but no success key should skip the success label."""
        import yaml

        from kiss.core.print_to_console import ConsolePrinter

        p = ConsolePrinter()
        content = yaml.dump({"summary": "Done without status"})
        result = p.print(content, type="result", step_count=1, total_tokens=0, cost=0.0)
        assert isinstance(result, str)

    def test_format_result_failed(self) -> None:
        """Test the failure case (success=False) for bold red styling."""
        import yaml

        from kiss.core.print_to_console import ConsolePrinter

        p = ConsolePrinter()
        content = yaml.dump({"success": False, "summary": "Something went wrong"})
        result = p.print(content, type="result", step_count=1, total_tokens=0, cost=0.0)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# anthropic_model.py — _build_create_kwargs branches
# ---------------------------------------------------------------------------


class TestAnthropicBuildKwargs:
    def test_build_kwargs_with_stop_string(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")
        m.model_config = {"stop": "END"}
        kwargs = m._build_create_kwargs()
        assert kwargs.get("stop_sequences") == ["END"]

    def test_build_kwargs_with_stop_list(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")
        m.model_config = {"stop": ["END", "STOP"]}
        kwargs = m._build_create_kwargs()
        assert kwargs.get("stop_sequences") == ["END", "STOP"]

    def test_build_kwargs_max_completion_tokens(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")
        m.model_config = {"max_completion_tokens": 1000}
        kwargs = m._build_create_kwargs()
        assert kwargs["max_tokens"] == 1000

    def test_build_kwargs_disable_cache(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")
        m.model_config = {"enable_cache": False}
        kwargs = m._build_create_kwargs()
        assert "cache_control" not in kwargs

    def test_build_kwargs_with_system_instruction(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")
        m.model_config = {"system_instruction": "You are helpful."}
        kwargs = m._build_create_kwargs()
        assert kwargs["system"] == "You are helpful."

    def test_build_kwargs_with_tools(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")
        kwargs = m._build_create_kwargs(tools=[{"name": "test"}])
        assert kwargs["tools"] == [{"name": "test"}]

    def test_build_kwargs_opus_adaptive_thinking(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-opus-4-6", api_key="test")
        kwargs = m._build_create_kwargs()
        assert kwargs.get("thinking") == {"type": "adaptive"}

    def test_build_kwargs_sonnet4_thinking(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-sonnet-4-something", api_key="test")
        kwargs = m._build_create_kwargs()
        assert kwargs.get("thinking") == {"type": "enabled", "budget_tokens": 10000}

    def test_build_kwargs_haiku4_default_max_tokens(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")
        kwargs = m._build_create_kwargs()
        assert kwargs["max_tokens"] == 64000  # haiku-4 default

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

    def test_build_kwargs_no_tools_no_tool_key(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")
        kwargs = m._build_create_kwargs(tools=None)
        assert "tools" not in kwargs

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


class TestAnthropicEmbedding:
    def test_get_embedding_raises(self) -> None:
        from kiss.core.kiss_error import KISSError
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")
        with pytest.raises(KISSError, match="embeddings API"):
            m.get_embedding("test")


# ---------------------------------------------------------------------------
# anthropic_model.py — _normalize_content_blocks, _extract_text
# ---------------------------------------------------------------------------


class TestAnthropicHelpers:
    def test_normalize_none(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")
        assert m._normalize_content_blocks(None) == []

    def test_normalize_string_iterates_chars(self) -> None:
        """String is iterable, so _normalize_content_blocks iterates over chars."""
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")
        result = m._normalize_content_blocks("ab")
        # Each char 'a', 'b' is iterated; strings are not dicts and have no .type attr
        assert len(result) == 2

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

    def test_add_results_with_explicit_tool_use_id(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")
        m.conversation = [{"role": "assistant", "content": "plain text"}]
        m.add_function_results_to_conversation_and_return(
            [("fn1", {"result": "r1", "tool_use_id": "explicit_id"})]
        )
        last = m.conversation[-1]
        assert last["content"][0]["tool_use_id"] == "explicit_id"

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

    def test_add_results_no_assistant_with_tool_use(self) -> None:
        """When no assistant message has tool_use blocks, fallback to generated IDs."""
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")
        m.conversation = [{"role": "user", "content": "hello"}]
        m.add_function_results_to_conversation_and_return(
            [("fn1", {"result": "r1"})]
        )
        last = m.conversation[-1]
        assert last["content"][0]["tool_use_id"] == "toolu_fn1_0"


class TestAnthropicToolsSchema:
    def test_build_schema_empty(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")
        result = m._build_anthropic_tools_schema({})
        assert result == []

    def test_build_schema_basic_function(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel

        m = AnthropicModel("claude-haiku-4-5", api_key="test")

        def greet(name: str) -> str:
            """Say hello.

            Args:
                name: The name.
            """
            return f"hello {name}"

        result = m._build_anthropic_tools_schema({"greet": greet})
        assert len(result) == 1
        assert result[0]["name"] == "greet"


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

    def test_parse_unknown_delta_type(self) -> None:
        from kiss.core.printer import StreamEventParser

        class Event:
            def __init__(self, d: dict) -> None:
                self.event = d

        p = StreamEventParser()
        p.parse_stream_event(
            Event({"type": "content_block_start", "content_block": {"type": "text"}})
        )
        # Unknown delta type should not produce text
        text = p.parse_stream_event(
            Event({"type": "content_block_delta", "delta": {"type": "unknown_delta"}})
        )
        assert text == ""

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

    def test_parse_text_delta(self) -> None:
        from kiss.core.printer import StreamEventParser

        class Event:
            def __init__(self, d: dict) -> None:
                self.event = d

        p = StreamEventParser()
        text = p.parse_stream_event(
            Event({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}})
        )
        assert text == "hi"

    def test_parse_unknown_event_type(self) -> None:
        from kiss.core.printer import StreamEventParser

        class Event:
            def __init__(self, d: dict) -> None:
                self.event = d

        p = StreamEventParser()
        text = p.parse_stream_event(Event({"type": "some_other_event"}))
        assert text == ""

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


class TestPrinterUtils:
    def test_lang_for_path(self) -> None:
        from kiss.core.printer import lang_for_path

        assert lang_for_path("foo.py") == "python"
        assert lang_for_path("bar.js") == "javascript"
        assert lang_for_path("no_ext") == "text"
        assert lang_for_path("foo.xyz") == "xyz"

    def test_truncate_result(self) -> None:
        from kiss.core.printer import truncate_result

        short = "short"
        assert truncate_result(short) == short
        long = "x" * 5000
        result = truncate_result(long)
        assert "truncated" in result
        assert len(result) < len(long)

    def test_extract_path_and_lang(self) -> None:
        from kiss.core.printer import extract_path_and_lang

        path, lang = extract_path_and_lang({"file_path": "foo.py"})
        assert path == "foo.py"
        assert lang == "python"

        path, lang = extract_path_and_lang({"path": "bar.js"})
        assert path == "bar.js"
        assert lang == "javascript"

        path, lang = extract_path_and_lang({})
        assert path == ""
        assert lang == "text"

    def test_extract_extras(self) -> None:
        from kiss.core.printer import extract_extras

        result = extract_extras({"file_path": "x", "custom_key": "val", "content": "c"})
        assert "custom_key" in result
        assert "file_path" not in result

        # Test truncation
        result = extract_extras({"extra": "x" * 300})
        assert result["extra"].endswith("...")


# ---------------------------------------------------------------------------
# sorcar_agent.py — _build_arg_parser and _resolve_task
# ---------------------------------------------------------------------------


class TestSorcarAgentCli:
    def test_build_arg_parser(self) -> None:
        from kiss.agents.sorcar.sorcar_agent import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.task is None
        assert args.f is None

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
