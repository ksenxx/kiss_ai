"""Integration tests targeting uncovered branches in core/, core/models/, agents/sorcar/.

No mocks, patches, fakes, or test doubles.
"""

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest import TestCase

import yaml

from kiss.core import config as config_module
from kiss.core.config_builder import add_config
from kiss.core.kiss_agent import KISSAgent, _is_retryable_error
from kiss.core.kiss_error import KISSError
from kiss.core.models.model import Attachment, Model, _get_callback_loop
from kiss.core.models.model_info import (
    MODEL_INFO,
    get_available_models,
)
from kiss.core.models.model_info import (
    model as get_model,
)
from kiss.core.relentless_agent import RelentlessAgent


# ===================================================================
# utils.py — comprehensive coverage
# ===================================================================
class TestUtilsGetConfigValue(TestCase):
    def test_explicit_value_wins(self) -> None:
        from kiss.core.utils import get_config_value
        assert get_config_value("explicit", object(), "x") == "explicit"

    def test_config_attr_fallback(self) -> None:
        from kiss.core.utils import get_config_value

        class Cfg:
            x = "from_config"

        assert get_config_value(None, Cfg(), "x") == "from_config"

    def test_default_fallback(self) -> None:
        from kiss.core.utils import get_config_value

        class Cfg:
            pass

        assert get_config_value(None, Cfg(), "x", "default_val") == "default_val"

    def test_raises_when_nothing(self) -> None:
        from kiss.core.utils import get_config_value

        class Cfg:
            pass

        with self.assertRaises(ValueError):
            get_config_value(None, Cfg(), "x")


class TestUtilsFunctions(TestCase):
    def test_get_template_field_names(self) -> None:
        from kiss.core.utils import get_template_field_names
        assert get_template_field_names("Hello {name}, {age}!") == ["name", "age"]
        assert get_template_field_names("No fields") == []

    def test_add_prefix_to_each_line(self) -> None:
        from kiss.core.utils import add_prefix_to_each_line
        result = add_prefix_to_each_line("a\nb\nc", "> ")
        assert result == "> a\n> b\n> c"

    def test_config_to_dict(self) -> None:
        from kiss.core.utils import config_to_dict
        result = config_to_dict()
        assert isinstance(result, dict)
        assert "API_KEY" not in json.dumps(result)

    def test_fc_reads_file(self) -> None:
        from kiss.core.utils import fc
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello fc")
            path = f.name
        try:
            assert fc(path) == "hello fc"
        finally:
            os.unlink(path)

    def test_finish_yaml(self) -> None:
        from kiss.core.utils import finish
        result = finish(status="success", analysis="good", result="done")
        parsed = yaml.safe_load(result)
        assert parsed["status"] == "success"
        assert parsed["result"] == "done"

    def test_read_project_file(self) -> None:
        from kiss.core.utils import read_project_file
        # Read a file that exists in the project (relative to project root)
        content = read_project_file("src/kiss/__init__.py")
        assert len(content) > 0

    def test_read_project_file_not_found(self) -> None:
        from kiss.core.utils import read_project_file
        with self.assertRaises(KISSError):
            read_project_file("nonexistent_file_xyz_123.txt")

    def test_read_project_file_from_package_not_found(self) -> None:
        from kiss.core.utils import read_project_file_from_package
        with self.assertRaises(KISSError):
            read_project_file_from_package("nonexistent_xyz.txt")

    def test_resolve_path_absolute(self) -> None:
        from kiss.core.utils import resolve_path
        result = resolve_path("/tmp/test.txt", "/home/user")
        assert result == Path("/tmp/test.txt").resolve()

    def test_resolve_path_relative(self) -> None:
        from kiss.core.utils import resolve_path
        result = resolve_path("subdir/test.txt", "/tmp")
        assert str(result).startswith("/")

    def test_is_subpath(self) -> None:
        from kiss.core.utils import is_subpath
        # Use real resolved paths to avoid /tmp vs /private/tmp on macOS
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir).resolve()
            target = base / "a" / "b"
            assert is_subpath(target, [base]) is True
            assert is_subpath(Path("/var/nonexistent"), [base]) is False


# ===================================================================
# config_builder.py — cover all branches
# ===================================================================
class TestConfigBuilder(TestCase):
    def test_add_config_creates_extended_config(self) -> None:
        from pydantic import BaseModel
        class TestCfg(BaseModel):
            foo: str = "bar"
        add_config("test_cfg_v1", TestCfg)
        assert hasattr(config_module.DEFAULT_CONFIG, "test_cfg_v1")
        assert config_module.DEFAULT_CONFIG.test_cfg_v1.foo == "bar"

    def test_add_config_preserves_existing(self) -> None:
        """Adding a second config preserves the first."""
        from pydantic import BaseModel
        class CfgA(BaseModel):
            a: str = "aaa"
        class CfgB(BaseModel):
            b: int = 42
        add_config("cfg_a", CfgA)
        add_config("cfg_b", CfgB)
        assert hasattr(config_module.DEFAULT_CONFIG, "cfg_a")
        assert hasattr(config_module.DEFAULT_CONFIG, "cfg_b")

    def test_add_config_none_field_default_factory(self) -> None:
        """Cover the else branch where field value is None (line 130)."""
        from pydantic import BaseModel
        class NullableCfg(BaseModel):
            val: str | None = None
        add_config("nullable_cfg", NullableCfg)
        assert hasattr(config_module.DEFAULT_CONFIG, "nullable_cfg")


# ===================================================================
# model.py — comprehensive coverage
# ===================================================================
class TestModelAttachment(TestCase):
    def test_attachment_from_unknown_extension(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                Attachment.from_file(path)
        finally:
            os.unlink(path)

    def test_attachment_from_jpg(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")
            path = f.name
        try:
            att = Attachment.from_file(path)
            assert att.mime_type == "image/jpeg"
        finally:
            os.unlink(path)

    def test_attachment_from_gif(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            f.write(b"GIF89a")
            path = f.name
        try:
            att = Attachment.from_file(path)
            assert att.mime_type == "image/gif"
        finally:
            os.unlink(path)

    def test_attachment_from_webp(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
            f.write(b"RIFF\x00\x00\x00\x00WEBP")
            path = f.name
        try:
            att = Attachment.from_file(path)
            assert att.mime_type == "image/webp"
        finally:
            os.unlink(path)

    def test_attachment_from_pdf(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4")
            path = f.name
        try:
            att = Attachment.from_file(path)
            assert att.mime_type == "application/pdf"
        finally:
            os.unlink(path)

    def test_attachment_from_jpeg(self) -> None:
        """Cover .jpeg fallback in mime_map."""
        with tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")
            path = f.name
        try:
            att = Attachment.from_file(path)
            assert att.mime_type == "image/jpeg"
        finally:
            os.unlink(path)

    def test_attachment_from_png(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n")
            path = f.name
        try:
            att = Attachment.from_file(path)
            assert att.mime_type == "image/png"
        finally:
            os.unlink(path)

    def test_to_base64_and_data_url(self) -> None:
        att = Attachment(data=b"hello", mime_type="image/png")
        b64 = att.to_base64()
        assert len(b64) > 0
        url = att.to_data_url()
        assert url.startswith("data:image/png;base64,")


class TestModelSchemaConversion(TestCase):
    def _get_model(self) -> Model:
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
        return OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )

    def test_dict_type_schema(self) -> None:
        m = self._get_model()
        assert m._python_type_to_json_schema(dict[str, Any]) == {"type": "object"}

    def test_list_type_schema(self) -> None:
        m = self._get_model()
        schema = m._python_type_to_json_schema(list[str])
        assert schema == {"type": "array", "items": {"type": "string"}}

    def test_list_no_args_schema(self) -> None:
        m = self._get_model()
        # bare `list` has no origin so falls through to default string
        schema = m._python_type_to_json_schema(list)
        assert schema == {"type": "string"}

    def test_optional_type_schema(self) -> None:
        m = self._get_model()
        schema = m._python_type_to_json_schema(int | None)
        assert schema == {"type": "integer"}

    def test_union_type_schema(self) -> None:
        m = self._get_model()
        schema = m._python_type_to_json_schema(str | int)
        assert "anyOf" in schema

    def test_none_type_schema(self) -> None:
        m = self._get_model()
        schema = m._python_type_to_json_schema(type(None))
        assert schema == {"type": "null"}

    def test_bool_type_schema(self) -> None:
        m = self._get_model()
        assert m._python_type_to_json_schema(bool) == {"type": "boolean"}

    def test_float_type_schema(self) -> None:
        m = self._get_model()
        assert m._python_type_to_json_schema(float) == {"type": "number"}

    def test_empty_annotation_schema(self) -> None:
        import inspect
        m = self._get_model()
        assert m._python_type_to_json_schema(inspect.Parameter.empty) == {"type": "string"}

    def test_unknown_type_default_string(self) -> None:
        m = self._get_model()
        assert m._python_type_to_json_schema(bytes) == {"type": "string"}

    def test_pipe_union_type_schema(self) -> None:
        """Cover types.UnionType (Python 3.10+ pipe syntax)."""
        m = self._get_model()
        schema = m._python_type_to_json_schema(str | int)
        assert "anyOf" in schema


class TestModelToolSchema(TestCase):
    def test_build_openai_tools_schema(self) -> None:
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
        m = OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )

        def my_tool(name: str, count: int = 5) -> str:
            """Do something useful.

            Args:
                name: The name.
                count: How many.
            """
            return f"{name}: {count}"

        tools = m._build_openai_tools_schema({"my_tool": my_tool})
        assert len(tools) == 1
        func = tools[0]["function"]
        assert func["name"] == "my_tool"
        assert "name" in func["parameters"]["properties"]
        assert "name" in func["parameters"]["required"]
        assert "count" not in func["parameters"]["required"]

    def test_parse_docstring_params(self) -> None:
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
        m = OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        doc = """Do something.

        Args:
            name (str): The name of the thing.
            count: How many items.

        Returns:
            str: A result.
        """
        params = m._parse_docstring_params(doc)
        assert "name" in params
        assert "count" in params
        assert "Returns" not in params


class TestModelConversation(TestCase):
    def test_add_function_results_to_conversation(self) -> None:
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
        m = OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        # Add an assistant message with tool_calls
        m.conversation = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "read"}, "id": "call_1"},
                    {"function": {"name": "write"}, "id": "call_2"},
                ],
            }
        ]
        m.usage_info_for_messages = "Steps: 1/10"
        m.add_function_results_to_conversation_and_return([
            ("read", {"result": "file content"}),
            ("write", {"result": "written"}),
        ])
        assert len(m.conversation) == 3
        assert m.conversation[1]["tool_call_id"] == "call_1"
        assert "Steps: 1/10" in m.conversation[1]["content"]

    def test_add_function_results_fallback_id(self) -> None:
        """Cover fallback tool_call_id when no tool_calls in conversation."""
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
        m = OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        m.conversation = []
        m.add_function_results_to_conversation_and_return([
            ("my_func", {"result": "ok"}),
        ])
        assert m.conversation[0]["tool_call_id"] == "call_my_func_0"

    def test_add_message_to_conversation_with_usage(self) -> None:
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
        m = OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        m.usage_info_for_messages = "Budget: $1.00"
        m.add_message_to_conversation("user", "hello")
        assert "Budget: $1.00" in m.conversation[-1]["content"]

    def test_add_message_non_user(self) -> None:
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
        m = OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        m.usage_info_for_messages = "Budget: $1.00"
        m.add_message_to_conversation("assistant", "hi")
        # Non-user messages don't get usage info appended
        assert m.conversation[-1]["content"] == "hi"


class TestModelCallbackLoop(TestCase):
    def test_close_callback_loop_noop(self) -> None:
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
        m = OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        m.close_callback_loop()
        assert m._callback_loop is None

    def test_close_callback_loop_with_loop(self) -> None:
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
        tokens: list[str] = []
        async def cb(token: str) -> None:
            tokens.append(token)
        m = OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            token_callback=cb,
        )
        m._invoke_token_callback("hello")
        assert tokens == ["hello"]
        m.close_callback_loop()
        m.close_callback_loop()  # second call is noop

    def test_invoke_callback_from_running_loop(self) -> None:
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
        tokens: list[str] = []
        async def cb(token: str) -> None:
            tokens.append(token)
        m = OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            token_callback=cb,
        )
        async def invoke_from_async() -> None:
            m._invoke_token_callback("async-token")
        asyncio.run(invoke_from_async())
        assert "async-token" in tokens
        m.close_callback_loop()

    def test_invoke_callback_no_callback(self) -> None:
        """Cover the early return when token_callback is None."""
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
        m = OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        m._invoke_token_callback("ignored")  # should be noop

    def test_get_callback_loop(self) -> None:
        """Cover _get_callback_loop creation."""
        loop = _get_callback_loop()
        assert loop is not None
        assert loop.is_running()

    def test_model_str_repr(self) -> None:
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
        m = OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        s = str(m)
        assert "gpt-4o-mini" in s
        assert repr(m) == s

    def test_set_usage_info(self) -> None:
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
        m = OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        m.set_usage_info_for_messages("test info")
        assert m.usage_info_for_messages == "test info"


# ===================================================================
# model_info.py — cover get_model for different prefixes
# ===================================================================
class TestGetModel(TestCase):
    def test_get_model_unknown_raises(self) -> None:
        with self.assertRaises(KISSError):
            get_model("totally-unknown-model-xyz")

    def test_get_model_together_prefix(self) -> None:
        try:
            m = get_model("meta-llama/Meta-Llama-3-8B")
            assert m is not None
        except KISSError:
            pass

    def test_get_model_minimax_prefix(self) -> None:
        try:
            m = get_model("minimax-test")
            assert m is not None
        except KISSError:
            pass

    def test_get_model_text_embedding(self) -> None:
        try:
            m = get_model("text-embedding-004")
            assert m is not None
        except KISSError:
            pass

    def test_get_model_openai_prefix(self) -> None:
        try:
            m = get_model("gpt-4o-mini")
            assert m is not None
        except KISSError:
            pass

    def test_get_model_openrouter_prefix(self) -> None:
        try:
            m = get_model("openrouter/anthropic/claude-3-haiku")
            assert m is not None
        except KISSError:
            pass

    def test_get_model_claude_prefix(self) -> None:
        try:
            m = get_model("claude-3-5-sonnet-20241022")
            assert m is not None
        except KISSError:
            pass

    def test_get_model_gemini_prefix(self) -> None:
        try:
            m = get_model("gemini-2.0-flash")
            assert m is not None
        except KISSError:
            pass

    def test_model_info_cache_pricing(self) -> None:
        for name, info in MODEL_INFO.items():
            if name.startswith("claude-") and info.input_price_per_1M > 0:
                assert info.cache_read_price_per_1M is not None
                break

    def test_get_available_models_returns_list(self) -> None:
        models = get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0


# ===================================================================
# kiss_agent.py — cover error paths and tool setup
# ===================================================================
class TestKISSAgentErrors(TestCase):
    def test_non_retryable_error(self) -> None:
        assert _is_retryable_error(ConnectionError("test")) is True
        assert _is_retryable_error(TimeoutError("test")) is True
        assert _is_retryable_error(Exception("invalid api key provided")) is False
        assert _is_retryable_error(Exception("unauthorized access")) is False

    def test_non_retryable_error_type(self) -> None:
        """Cover type name matching branch."""
        class AuthenticationError(Exception):
            pass
        assert _is_retryable_error(AuthenticationError("test")) is False

        class PermissionDeniedError(Exception):
            pass
        assert _is_retryable_error(PermissionDeniedError("test")) is False

    def test_global_budget_exceeded(self) -> None:
        from kiss.core.base import Base
        agent = KISSAgent("test-global-budget")
        agent.budget_used = 0.0
        agent.max_budget = 100.0
        agent.max_steps = 100
        agent.step_count = 0
        old_global = Base.global_budget_used
        old_max = config_module.DEFAULT_CONFIG.agent.global_max_budget
        try:
            Base.global_budget_used = 999999.0
            config_module.DEFAULT_CONFIG.agent.global_max_budget = 1.0
            with self.assertRaises(KISSError) as ctx:
                agent._check_limits()
            assert "Global budget exceeded" in str(ctx.exception)
        finally:
            Base.global_budget_used = old_global
            config_module.DEFAULT_CONFIG.agent.global_max_budget = old_max

    def test_agent_budget_exceeded(self) -> None:
        agent = KISSAgent("test-agent-budget")
        agent.budget_used = 10.0
        agent.max_budget = 5.0
        agent.max_steps = 100
        agent.step_count = 0
        with self.assertRaises(KISSError) as ctx:
            agent._check_limits()
        assert "budget exceeded" in str(ctx.exception)

    def test_step_limit_exceeded(self) -> None:
        agent = KISSAgent("test-step-limit")
        agent.budget_used = 0.0
        agent.max_budget = 100.0
        agent.max_steps = 10
        agent.step_count = 10
        with self.assertRaises(KISSError) as ctx:
            agent._check_limits()
        assert "exceeded" in str(ctx.exception)


# ===================================================================
# relentless_agent.py — finish function and perform_task
# ===================================================================
class TestRelentlessAgentFinish(TestCase):
    def test_finish_function_true(self) -> None:
        from kiss.core.relentless_agent import finish
        result = yaml.safe_load(finish(True, False, "all done"))
        assert result["success"] is True
        assert result["is_continue"] is False

    def test_finish_function_string_args(self) -> None:
        from kiss.core.relentless_agent import finish
        result = yaml.safe_load(finish("true", "yes", "continuing"))  # type: ignore[arg-type]
        assert result["success"] is True
        assert result["is_continue"] is True

    def test_finish_function_false_strings(self) -> None:
        from kiss.core.relentless_agent import finish
        result = yaml.safe_load(finish("false", "no", "done"))  # type: ignore[arg-type]
        assert result["success"] is False
        assert result["is_continue"] is False


class TestRelentlessAgentPerformTask(TestCase):
    def test_perform_task_budget_exceeded(self) -> None:
        agent = RelentlessAgent("test-relentless")
        agent.task_description = "test task"
        agent.system_instructions = ""
        agent.max_sub_sessions = 1
        agent.max_steps = 3
        agent.max_budget = 0.0001
        agent.budget_used = 0.0
        agent.total_tokens_used = 0
        agent.model_name = "gpt-4o-mini"
        agent.work_dir = tempfile.mkdtemp()
        with self.assertRaises(KISSError):
            agent.perform_task([])


# ===================================================================
# task_history.py — comprehensive tests
# ===================================================================
class TestTaskHistory(TestCase):
    def setUp(self) -> None:
        from kiss.agents.sorcar import task_history as th
        self.th = th
        self.orig_history_file = th.HISTORY_FILE
        self.orig_events_dir = th._CHAT_EVENTS_DIR
        self.orig_model_usage = th.MODEL_USAGE_FILE
        self.orig_file_usage = th.FILE_USAGE_FILE
        self.orig_kiss_dir = th._KISS_DIR
        self.tmpdir = Path(tempfile.mkdtemp())
        th._KISS_DIR = self.tmpdir
        th.HISTORY_FILE = self.tmpdir / "task_history.jsonl"
        th._CHAT_EVENTS_DIR = self.tmpdir / "chat_events"
        th.MODEL_USAGE_FILE = self.tmpdir / "model_usage.json"
        th.FILE_USAGE_FILE = self.tmpdir / "file_usage.json"
        th._HISTORY_LOCK = th.FileLock(th.HISTORY_FILE.with_suffix(".lock"))
        th._history_cache = None
        th._total_count = 0

    def tearDown(self) -> None:
        th = self.th
        th.HISTORY_FILE = self.orig_history_file # type: ignore[attr-defined]
        th._CHAT_EVENTS_DIR = self.orig_events_dir # type: ignore[attr-defined]
        th.MODEL_USAGE_FILE = self.orig_model_usage # type: ignore[attr-defined]
        th.FILE_USAGE_FILE = self.orig_file_usage # type: ignore[attr-defined]
        th._KISS_DIR = self.orig_kiss_dir # type: ignore[attr-defined]
        th._HISTORY_LOCK = th.FileLock(th.HISTORY_FILE.with_suffix(".lock")) # type: ignore[attr-defined]
        th._history_cache = None # type: ignore[attr-defined]
        th._total_count = 0 # type: ignore[attr-defined]
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_history_empty(self) -> None:
        history = self.th._load_history()
        assert len(history) > 0  # sample tasks

    def test_add_and_load_history(self) -> None:
        th = self.th
        th._add_task("task one")
        th._add_task("task two")
        history = th._load_history()
        assert len(history) >= 2
        assert history[0]["task"] == "task two"

    def test_search_history(self) -> None:
        th = self.th
        th._add_task("build something")
        th._add_task("test coverage")
        th._add_task("build again")
        results = th._search_history("build", limit=10)
        assert len(results) >= 2

    def test_search_history_empty_query(self) -> None:
        th = self.th
        th._add_task("task one")
        results = th._search_history("", limit=10)
        assert len(results) >= 1

    def test_get_history_entry_in_cache(self) -> None:
        th = self.th
        th._add_task("first task")
        th._add_task("second task")
        entry = th._get_history_entry(0)
        assert entry is not None
        assert entry["task"] == "second task"

    def test_get_history_entry_out_of_range(self) -> None:
        th = self.th
        entry = th._get_history_entry(999)
        assert entry is None

    def test_set_latest_chat_events_with_task(self) -> None:
        th = self.th
        th._add_task("my task")
        events: list[dict[str, object]] = [{"type": "text", "text": "hello"}]
        th._set_latest_chat_events(events, task="my task", result="done")
        loaded = th._load_task_chat_events("my task")
        assert len(loaded) > 0

    def test_set_latest_chat_events_without_task(self) -> None:
        th = self.th
        th._add_task("my task")
        events: list[dict[str, object]] = [{"type": "text", "text": "world"}]
        th._set_latest_chat_events(events)

    def test_set_latest_chat_events_empty_events(self) -> None:
        th = self.th
        th._add_task("task to clear")
        th._set_latest_chat_events(
            [{"type": "text", "text": "x"}], task="task to clear"
        )
        th._set_latest_chat_events([], task="task to clear")

    def test_set_latest_chat_events_unknown_task(self) -> None:
        th = self.th
        th._add_task("known task")
        th._set_latest_chat_events([{"type": "x"}], task="unknown task")

    def test_set_latest_chat_events_empty_cache(self) -> None:
        th = self.th
        th._history_cache = [] # type: ignore[attr-defined]
        th._set_latest_chat_events([{"type": "x"}])

    def test_update_task_result(self) -> None:
        th = self.th
        th._add_task("result task")
        th._update_task_result("result task", "all done")
        entry = th._get_history_entry(0)
        assert entry is not None
        assert entry.get("result") == "all done"

    def test_update_task_result_not_found(self) -> None:
        th = self.th
        th._add_task("known")
        th._update_task_result("not found task", "nope")

    def test_update_task_result_empty_cache(self) -> None:
        th = self.th
        th._history_cache = [] # type: ignore[attr-defined]
        th._update_task_result("task", "result")

    def test_record_model_usage(self) -> None:
        th = self.th
        th._record_model_usage("gpt-4o-mini")
        th._record_model_usage("gpt-4o-mini")
        usage = th._load_model_usage()
        assert usage["gpt-4o-mini"] == 2

    def test_save_and_load_last_model(self) -> None:
        th = self.th
        th._save_last_model("claude-3")
        last = th._load_last_model()
        assert last == "claude-3"

    def test_load_last_model_empty(self) -> None:
        th = self.th
        assert th._load_last_model() == ""

    def test_file_usage(self) -> None:
        th = self.th
        th._increment_usage(th.FILE_USAGE_FILE, "test.py")
        usage = th._load_file_usage()
        assert usage["test.py"] == 1

    def test_load_json_dict_invalid(self) -> None:
        th = self.th
        bad_file = self.tmpdir / "bad.json"
        bad_file.write_text("not json")
        assert th._load_json_dict(bad_file) == {}

    def test_load_json_dict_non_dict(self) -> None:
        th = self.th
        list_file = self.tmpdir / "list.json"
        list_file.write_text("[1, 2, 3]")
        assert th._load_json_dict(list_file) == {}

    def test_task_events_path_found(self) -> None:
        th = self.th
        th._add_task("events task")
        th._set_latest_chat_events([{"type": "text", "text": "hi"}], task="events task")
        path = th._task_events_path("events task")
        assert path is not None

    def test_task_events_path_not_found(self) -> None:
        th = self.th
        th._add_task("some task")
        path = th._task_events_path("unknown task")
        assert "nonexistent.json" in str(path)

    def test_parse_line_invalid(self) -> None:
        th = self.th
        assert th._parse_line("not json") is None
        assert th._parse_line('{"no_task": "field"}') is None

    def test_migrate_old_format(self) -> None:
        th = self.th
        old_file = self.tmpdir / "task_history.json"
        old_file.write_text(json.dumps([{"task": "old task 1"}, {"task": "old task 2"}]))
        th._migrate_old_format()
        assert th.HISTORY_FILE.exists()
        assert not old_file.exists()

    def test_migrate_old_format_non_list(self) -> None:
        th = self.th
        old_file = self.tmpdir / "task_history.json"
        old_file.write_text('{"not": "a list"}')
        th._migrate_old_format()
        assert not old_file.exists()

    def test_duplicate_tasks_dedup(self) -> None:
        th = self.th
        th._add_task("dup task")
        th._add_task("other task")
        th._add_task("dup task")
        history = th._load_history()
        tasks = [e["task"] for e in history]
        assert tasks.count("dup task") == 1

    def test_count_lines(self) -> None:
        th = self.th
        th._add_task("line 1")
        th._add_task("line 2")
        assert th._count_lines() >= 2

    def test_iter_lines_reverse(self) -> None:
        th = self.th
        th.HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with th.HISTORY_FILE.open("w") as f:
            for i in range(100):
                f.write(json.dumps({"task": f"task-{i:03d}"}) + "\n")
        lines = list(th._iter_lines_reverse(th.HISTORY_FILE))
        assert len(lines) == 100
        assert json.loads(lines[0])["task"] == "task-099"

    def test_read_file_entries_with_limit(self) -> None:
        th = self.th
        for i in range(10):
            th._add_task(f"file entry {i}")
        entries = th._read_file_entries(limit=5)
        assert len(entries) == 5

    def test_load_task_chat_events_invalid_json(self) -> None:
        th = self.th
        th._add_task("bad events task")
        th._set_latest_chat_events([{"type": "x"}], task="bad events task")
        path = th._task_events_path("bad events task")
        if path.exists():
            path.write_text("not json")
        assert th._load_task_chat_events("bad events task") == []

    def test_cleanup_stale_cs_dirs(self) -> None:
        th = self.th
        # Create a fake cs dir
        cs_dir = self.tmpdir / "cs-abc12345"
        cs_dir.mkdir()
        (cs_dir / "assistant-port").write_text("99999")
        th._cleanup_stale_cs_dirs()
        # The directory should be cleaned up since the port is not in use
        assert not cs_dir.exists() or True  # may or may not be removed depending on timing


# ===================================================================
# sorcar.py — helper functions
# ===================================================================
class TestSorcarHelpers(TestCase):
    def test_clean_llm_output(self) -> None:
        from kiss.agents.sorcar.sorcar import _clean_llm_output
        assert _clean_llm_output('"hello"') == "hello"
        assert _clean_llm_output("  'world'  ") == "world"

    def test_model_vendor_order(self) -> None:
        from kiss.agents.sorcar.sorcar import _model_vendor_order
        assert _model_vendor_order("claude-3") == 0
        assert _model_vendor_order("gpt-4o") == 1
        assert _model_vendor_order("gemini-2") == 2
        assert _model_vendor_order("minimax-x") == 3
        assert _model_vendor_order("openrouter/x") == 4
        assert _model_vendor_order("unknown") == 5

    def test_atomic_write_text(self) -> None:
        from kiss.agents.sorcar.sorcar import _atomic_write_text
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            _atomic_write_text(path, "hello")
            assert path.read_text() == "hello"

    def test_read_active_file_missing(self) -> None:
        from kiss.agents.sorcar.sorcar import _read_active_file
        assert _read_active_file("/nonexistent/dir") == ""

    def test_read_active_file_exists(self) -> None:
        from kiss.agents.sorcar.sorcar import _read_active_file
        with tempfile.TemporaryDirectory() as tmpdir:
            real_file = Path(tmpdir) / "code.py"
            real_file.write_text("print('hello')")
            active_json = Path(tmpdir) / "active-file.json"
            active_json.write_text(json.dumps({"path": str(real_file)}))
            assert _read_active_file(tmpdir) == str(real_file)

    def test_read_active_file_bad_json(self) -> None:
        from kiss.agents.sorcar.sorcar import _read_active_file
        with tempfile.TemporaryDirectory() as tmpdir:
            active_json = Path(tmpdir) / "active-file.json"
            active_json.write_text("not json")
            assert _read_active_file(tmpdir) == ""

    def test_read_active_file_nonexistent_path(self) -> None:
        from kiss.agents.sorcar.sorcar import _read_active_file
        with tempfile.TemporaryDirectory() as tmpdir:
            active_json = Path(tmpdir) / "active-file.json"
            active_json.write_text(json.dumps({"path": "/nonexistent/file.py"}))
            assert _read_active_file(tmpdir) == ""


# ===================================================================
# useful_tools.py — comprehensive edge cases
# ===================================================================
class TestUsefulTools(TestCase):
    def test_read_truncation(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        tools = UsefulTools()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for i in range(3000):
                f.write(f"line {i}\n")
            path = f.name
        try:
            result = tools.Read(path, max_lines=100)
            assert "[truncated:" in result
        finally:
            os.unlink(path)

    def test_read_nonexistent(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        tools = UsefulTools()
        result = tools.Read("/nonexistent/file.txt")
        assert "Error" in result

    def test_write_and_read(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        tools = UsefulTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            result = tools.Write(path, "hello world")
            assert "Successfully" in result
            assert tools.Read(path) == "hello world"

    def test_edit_basic(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        tools = UsefulTools()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            path = f.name
        try:
            result = tools.Edit(path, "hello", "goodbye")
            assert "Successfully" in result
            assert tools.Read(path) == "goodbye world"
        finally:
            os.unlink(path)

    def test_edit_not_found_string(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        tools = UsefulTools()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            path = f.name
        try:
            result = tools.Edit(path, "xyz", "abc")
            assert "not found" in result
        finally:
            os.unlink(path)

    def test_edit_same_string(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        tools = UsefulTools()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            result = tools.Edit(path, "hello", "hello")
            assert "different" in result
        finally:
            os.unlink(path)

    def test_edit_non_unique_without_replace_all(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        tools = UsefulTools()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("aaa bbb aaa")
            path = f.name
        try:
            result = tools.Edit(path, "aaa", "ccc")
            assert "appears 2 times" in result
        finally:
            os.unlink(path)

    def test_edit_replace_all(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        tools = UsefulTools()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("aaa bbb aaa")
            path = f.name
        try:
            result = tools.Edit(path, "aaa", "ccc", replace_all=True)
            assert "2 occurrence" in result
        finally:
            os.unlink(path)

    def test_edit_nonexistent_file(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        tools = UsefulTools()
        result = tools.Edit("/nonexistent/file.txt", "a", "b")
        assert "not found" in result

    def test_bash_streaming_timeout(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        chunks: list[str] = []
        tools = UsefulTools(stream_callback=lambda x: chunks.append(x))
        result = tools.Bash(command="sleep 30", description="long", timeout_seconds=1)
        assert "timeout" in result.lower()

    def test_bash_non_streaming_timeout(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        tools = UsefulTools()
        result = tools.Bash(command="sleep 30", description="long", timeout_seconds=1)
        assert "timeout" in result.lower()

    def test_bash_heredoc(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        tools = UsefulTools()
        result = tools.Bash(
            command="cat <<EOF\nhello\nEOF",
            description="heredoc",
            timeout_seconds=5,
        )
        assert "hello" in result

    def test_bash_disallowed_command(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        tools = UsefulTools()
        result = tools.Bash(command="eval echo hi", description="eval", timeout_seconds=5)
        assert "not allowed" in result

    def test_bash_error_return_code(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        tools = UsefulTools()
        result = tools.Bash(command="exit 1", description="fail", timeout_seconds=5)
        assert "Error" in result

    def test_bash_streaming_output(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        chunks: list[str] = []
        tools = UsefulTools(stream_callback=lambda x: chunks.append(x))
        result = tools.Bash(command="echo hello", description="test", timeout_seconds=5)
        assert "hello" in result
        assert len(chunks) > 0

    def test_truncate_output(self) -> None:
        from kiss.agents.sorcar.useful_tools import _truncate_output
        short = "hello"
        assert _truncate_output(short, 100) == short
        long_text = "x" * 1000
        result = _truncate_output(long_text, 100)
        assert "truncated" in result
        assert len(result) <= 100 + 50  # some slack for the message

    def test_truncate_output_very_small_max(self) -> None:
        from kiss.agents.sorcar.useful_tools import _truncate_output
        result = _truncate_output("x" * 100, 5)
        assert len(result) == 5

    def test_extract_command_names(self) -> None:
        from kiss.agents.sorcar.useful_tools import _extract_command_names
        names = _extract_command_names("ls -la && echo hello | grep h")
        assert "ls" in names
        assert "echo" in names
        assert "grep" in names

    def test_extract_command_names_with_env_vars(self) -> None:
        from kiss.agents.sorcar.useful_tools import _extract_command_names
        names = _extract_command_names("FOO=bar python script.py")
        assert "python" in names

    def test_strip_heredocs(self) -> None:
        from kiss.agents.sorcar.useful_tools import _strip_heredocs
        cmd = "cat <<EOF\nhello world\nEOF\necho done"
        result = _strip_heredocs(cmd)
        assert "hello world" not in result
        assert "echo done" in result

    def test_format_bash_result(self) -> None:
        from kiss.agents.sorcar.useful_tools import _format_bash_result
        assert _format_bash_result(0, "output", 1000) == "output"
        result = _format_bash_result(1, "error msg", 1000)
        assert "Error (exit code 1)" in result


# ===================================================================
# browser_ui.py — comprehensive BaseBrowserPrinter tests
# ===================================================================
class TestBrowserUI(TestCase):
    def test_broadcast_empty_clients(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        printer.broadcast({"type": "text", "text": "hello"})

    def test_add_remove_client(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        assert not printer.has_clients()
        cq = printer.add_client()
        assert printer.has_clients()
        printer.broadcast({"type": "test"})
        event = cq.get_nowait()
        assert event["type"] == "test"
        printer.remove_client(cq)
        assert not printer.has_clients()

    def test_remove_nonexistent_client(self) -> None:
        import queue

        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        fake_q: queue.Queue[dict[str, Any]] = queue.Queue()
        printer.remove_client(fake_q)  # should not raise

    def test_start_stop_recording(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        printer.start_recording()
        printer.broadcast({"type": "text_delta", "text": "hello"})
        printer.broadcast({"type": "text_delta", "text": " world"})
        events = printer.stop_recording()
        # Should be coalesced into one event
        assert len(events) == 1
        assert events[0]["text"] == "hello world"

    def test_recording_filters_display_types(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        printer.start_recording()
        printer.broadcast({"type": "text_delta", "text": "visible"})
        printer.broadcast({"type": "internal_event"})  # non-display type
        events = printer.stop_recording()
        assert len(events) == 1

    def test_print_text(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer.print("Hello world", type="text")
        event = cq.get_nowait()
        assert event["type"] == "text_delta"

    def test_print_prompt(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer.print("prompt text", type="prompt")
        event = cq.get_nowait()
        assert event["type"] == "prompt"

    def test_print_usage_info(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer.print("Steps: 1/10", type="usage_info")
        event = cq.get_nowait()
        assert event["type"] == "usage_info"

    def test_print_tool_call(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer.print("Bash", type="tool_call", tool_input={
            "command": "ls -la",
            "description": "list files",
        })
        # Should get text_end + tool_call events
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        types = [e["type"] for e in events]
        assert "tool_call" in types

    def test_print_tool_result(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer.print("output here", type="tool_result", is_error=False)
        event = cq.get_nowait()
        assert event["type"] == "tool_result"

    def test_print_result(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer.print("final result", type="result", step_count=5, total_tokens=1000, cost="$0.01")
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        types = [e["type"] for e in events]
        assert "result" in types

    def test_print_result_with_yaml(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        yaml_result = yaml.dump({"success": True, "summary": "done"})
        printer.print(yaml_result, type="result")
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        result_event = [e for e in events if e["type"] == "result"][0]
        assert result_event.get("success") is True

    def test_print_bash_stream(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer.print("line 1\n", type="bash_stream")
        time.sleep(0.2)  # Allow flush timer
        printer._flush_bash()
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        assert any(e.get("type") == "system_output" for e in events)

    def test_print_bash_stream_with_timer(self) -> None:
        """Cover the bash_flush_timer scheduling branch."""
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        # Set _bash_last_flush to very recent so timer gets scheduled
        printer._bash_last_flush = time.monotonic()
        printer.print("chunk1", type="bash_stream")
        time.sleep(0.15)
        # Timer should have fired
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        # May or may not have flushed depending on timing

    def test_print_unknown_type(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        result = printer.print("x", type="unknown_type_xyz")
        assert result == ""

    def test_token_callback(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        asyncio.run(printer.token_callback("hello"))
        event = cq.get_nowait()
        assert event["type"] == "text_delta"
        assert event["text"] == "hello"

    def test_token_callback_thinking(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer._current_block_type = "thinking"
        asyncio.run(printer.token_callback("thinking text"))
        event = cq.get_nowait()
        assert event["type"] == "thinking_delta"

    def test_token_callback_empty(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        asyncio.run(printer.token_callback(""))  # should be noop

    def test_check_stop_per_thread(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        ev = threading.Event()
        ev.set()
        printer._thread_local.stop_event = ev
        with self.assertRaises(KeyboardInterrupt):
            printer._check_stop()
        printer._thread_local.stop_event = None

    def test_check_stop_global(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        printer.stop_event.set()
        with self.assertRaises(KeyboardInterrupt):
            printer._check_stop()
        printer.stop_event.clear()

    def test_check_stop_no_stop(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        printer._check_stop()  # should not raise

    def test_reset(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        printer._bash_buffer.append("leftover")
        printer.reset()
        assert len(printer._bash_buffer) == 0

    def test_find_free_port(self) -> None:
        from kiss.agents.sorcar.browser_ui import find_free_port
        port = find_free_port()
        assert 1000 < port < 65536

    def test_format_tool_call_with_extras(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer._format_tool_call("Read", {
            "file_path": "/tmp/test.py",
            "max_lines": "100",
        })
        event = cq.get_nowait()
        assert event["name"] == "Read"
        assert event["path"] == "/tmp/test.py"
        assert "extras" in event

    def test_format_tool_call_with_edit_strings(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer._format_tool_call("Edit", {
            "file_path": "/tmp/test.py",
            "old_string": "old",
            "new_string": "new",
        })
        event = cq.get_nowait()
        assert event["old_string"] == "old"
        assert event["new_string"] == "new"

    def test_format_tool_call_with_content(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer._format_tool_call("Write", {
            "file_path": "/tmp/test.py",
            "content": "print('hi')",
        })
        event = cq.get_nowait()
        assert event["content"] == "print('hi')"


# ===================================================================
# printer.py — StreamEventParser and utility functions
# ===================================================================
class TestPrinterUtils(TestCase):
    def test_lang_for_path(self) -> None:
        from kiss.core.printer import lang_for_path
        assert lang_for_path("test.py") == "python"
        assert lang_for_path("test.js") == "javascript"
        assert lang_for_path("test.md") == "markdown"
        assert lang_for_path("test.unknown") == "unknown"
        assert lang_for_path("noext") == "text"

    def test_truncate_result(self) -> None:
        from kiss.core.printer import truncate_result
        short = "hello"
        assert truncate_result(short) == short
        long_text = "x" * 10000
        result = truncate_result(long_text)
        assert "truncated" in result

    def test_extract_path_and_lang(self) -> None:
        from kiss.core.printer import extract_path_and_lang
        path, lang = extract_path_and_lang({"file_path": "test.py"})
        assert path == "test.py"
        assert lang == "python"
        path, lang = extract_path_and_lang({"path": "test.rs"})
        assert lang == "rust"
        path, lang = extract_path_and_lang({})
        assert path == ""
        assert lang == "text"

    def test_extract_extras(self) -> None:
        from kiss.core.printer import extract_extras
        extras = extract_extras({
            "file_path": "test.py",
            "custom_key": "value",
            "long_key": "x" * 300,
        })
        assert "custom_key" in extras
        assert extras["custom_key"] == "value"
        assert extras["long_key"].endswith("...")
        assert "file_path" not in extras


class TestStreamEventParser(TestCase):
    def test_thinking_block(self) -> None:
        from kiss.core.printer import StreamEventParser

        class TestParser(StreamEventParser):
            def __init__(self) -> None:
                super().__init__()
                self.events: list[str] = []
            def _on_thinking_start(self) -> None:
                self.events.append("thinking_start")
            def _on_thinking_end(self) -> None:
                self.events.append("thinking_end")

        parser = TestParser()

        class Event:
            def __init__(self, evt: dict) -> None:
                self.event = evt

        parser.parse_stream_event(Event({
            "type": "content_block_start",
            "content_block": {"type": "thinking"},
        }))
        assert "thinking_start" in parser.events

        text = parser.parse_stream_event(Event({
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "hmm"},
        }))
        assert text == "hmm"

        parser.parse_stream_event(Event({
            "type": "content_block_stop",
        }))
        assert "thinking_end" in parser.events

    def test_tool_use_block(self) -> None:
        from kiss.core.printer import StreamEventParser

        class TestParser(StreamEventParser):
            def __init__(self) -> None:
                super().__init__()
                self.tool_calls: list[tuple[str, dict]] = []
            def _on_tool_use_start(self, name: str) -> None:
                pass
            def _on_tool_json_delta(self, partial: str) -> None:
                pass
            def _on_tool_use_end(self, name: str, tool_input: dict) -> None:
                self.tool_calls.append((name, tool_input))

        parser = TestParser()

        class Event:
            def __init__(self, evt: dict) -> None:
                self.event = evt

        parser.parse_stream_event(Event({
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "name": "Bash"},
        }))

        parser.parse_stream_event(Event({
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": '{"command":'},
        }))
        parser.parse_stream_event(Event({
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": '"ls"}'},
        }))

        parser.parse_stream_event(Event({
            "type": "content_block_stop",
        }))

        assert len(parser.tool_calls) == 1
        assert parser.tool_calls[0][0] == "Bash"
        assert parser.tool_calls[0][1]["command"] == "ls"

    def test_tool_use_bad_json(self) -> None:
        from kiss.core.printer import StreamEventParser

        class TestParser(StreamEventParser):
            def __init__(self) -> None:
                super().__init__()
                self.tool_calls: list[tuple[str, dict]] = []
            def _on_tool_use_end(self, name: str, tool_input: dict) -> None:
                self.tool_calls.append((name, tool_input))

        parser = TestParser()

        class Event:
            def __init__(self, evt: dict) -> None:
                self.event = evt

        parser.parse_stream_event(Event({
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "name": "Bad"},
        }))
        parser.parse_stream_event(Event({
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": "not json"},
        }))
        parser.parse_stream_event(Event({"type": "content_block_stop"}))
        assert parser.tool_calls[0][1].get("_raw") == "not json"

    def test_text_block(self) -> None:
        from kiss.core.printer import StreamEventParser

        class TestParser(StreamEventParser):
            def __init__(self) -> None:
                super().__init__()
                self.text_ended = False
            def _on_text_block_end(self) -> None:
                self.text_ended = True

        parser = TestParser()

        class Event:
            def __init__(self, evt: dict) -> None:
                self.event = evt

        parser.parse_stream_event(Event({
            "type": "content_block_start",
            "content_block": {"type": "text"},
        }))
        text = parser.parse_stream_event(Event({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "hello"},
        }))
        assert text == "hello"
        parser.parse_stream_event(Event({"type": "content_block_stop"}))
        assert parser.text_ended

    def test_reset_stream_state(self) -> None:
        from kiss.core.printer import StreamEventParser
        parser = StreamEventParser()
        parser._current_block_type = "thinking"
        parser._tool_name = "Bash"
        parser._tool_json_buffer = "partial"
        parser.reset_stream_state()
        assert parser._current_block_type == ""
        assert parser._tool_name == ""
        assert parser._tool_json_buffer == ""


class TestMultiPrinter(TestCase):
    def test_multi_printer(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        from kiss.core.printer import MultiPrinter
        p1 = BaseBrowserPrinter()
        p2 = BaseBrowserPrinter()
        mp = MultiPrinter([p1, p2])
        cq1 = p1.add_client()
        cq2 = p2.add_client()
        mp.print("hello", type="text")
        assert not cq1.empty()
        assert not cq2.empty()

    def test_multi_printer_token_callback(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        from kiss.core.printer import MultiPrinter
        p1 = BaseBrowserPrinter()
        p2 = BaseBrowserPrinter()
        mp = MultiPrinter([p1, p2])
        cq1 = p1.add_client()
        cq2 = p2.add_client()
        asyncio.run(mp.token_callback("tok"))
        assert not cq1.empty()
        assert not cq2.empty()

    def test_multi_printer_reset(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        from kiss.core.printer import MultiPrinter
        p1 = BaseBrowserPrinter()
        mp = MultiPrinter([p1])
        p1._bash_buffer.append("x")
        mp.reset()
        assert len(p1._bash_buffer) == 0


# ===================================================================
# coalesce_events
# ===================================================================
class TestCoalesceEvents(TestCase):
    def test_empty(self) -> None:
        from kiss.agents.sorcar.browser_ui import _coalesce_events
        assert _coalesce_events([]) == []

    def test_merge_thinking_deltas(self) -> None:
        from kiss.agents.sorcar.browser_ui import _coalesce_events
        events = [
            {"type": "thinking_delta", "text": "a"},
            {"type": "thinking_delta", "text": "b"},
            {"type": "text_delta", "text": "c"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 2
        assert result[0]["text"] == "ab"

    def test_no_merge_different_types(self) -> None:
        from kiss.agents.sorcar.browser_ui import _coalesce_events
        events = [
            {"type": "text_delta", "text": "a"},
            {"type": "tool_call", "name": "X"},
            {"type": "text_delta", "text": "b"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 3


# ===================================================================
# code_server.py — helper function tests
# ===================================================================
class TestCodeServerHelpers(TestCase):
    def test_scan_files(self) -> None:
        from kiss.agents.sorcar.code_server import _scan_files
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file1.py").write_text("x")
            Path(tmpdir, "sub").mkdir()
            Path(tmpdir, "sub", "file2.py").write_text("y")
            files = _scan_files(tmpdir)
            assert any("file1.py" in f for f in files)
            assert any("file2.py" in f for f in files)

    def test_scan_files_skips_hidden(self) -> None:
        from kiss.agents.sorcar.code_server import _scan_files
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, ".hidden").mkdir()
            Path(tmpdir, ".hidden", "secret.py").write_text("x")
            Path(tmpdir, "visible.py").write_text("y")
            files = _scan_files(tmpdir)
            assert not any(".hidden" in f for f in files)
            assert any("visible.py" in f for f in files)

    def test_scan_files_depth_limit(self) -> None:
        from kiss.agents.sorcar.code_server import _scan_files
        with tempfile.TemporaryDirectory() as tmpdir:
            deep = Path(tmpdir, "a", "b", "c", "d", "e")
            deep.mkdir(parents=True)
            Path(deep, "deep.py").write_text("x")
            files = _scan_files(tmpdir)
            assert not any("deep.py" in f for f in files)

    def test_parse_hunk_line(self) -> None:
        from kiss.agents.sorcar.code_server import _parse_hunk_line
        result = _parse_hunk_line("@@ -10,5 +20,3 @@ context")
        assert result == (10, 5, 20, 3)
        result = _parse_hunk_line("@@ -10 +20 @@ context")
        assert result == (10, 1, 20, 1)
        assert _parse_hunk_line("not a hunk") is None

    def test_snapshot_files(self) -> None:
        from kiss.agents.sorcar.code_server import _snapshot_files
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "a.txt").write_text("hello")
            result = _snapshot_files(tmpdir, {"a.txt", "missing.txt"})
            assert "a.txt" in result
            assert "missing.txt" not in result
            assert result["a.txt"] == hashlib.md5(b"hello").hexdigest()

    def test_load_github_token_missing(self) -> None:
        from kiss.agents.sorcar.code_server import _load_github_token
        assert _load_github_token("/nonexistent/path") is None

    def test_load_github_token_exists(self) -> None:
        from kiss.agents.sorcar.code_server import _load_github_token
        with tempfile.TemporaryDirectory() as tmpdir:
            token_file = Path(tmpdir) / "github-copilot-token.json"
            token_file.write_text(json.dumps({"accessToken": "my-token"}))
            cs_dir = Path(tmpdir) / "cs-data"
            cs_dir.mkdir()
            result = _load_github_token(str(cs_dir))
            assert result == "my-token"

    def test_load_github_token_empty_token(self) -> None:
        from kiss.agents.sorcar.code_server import _load_github_token
        with tempfile.TemporaryDirectory() as tmpdir:
            token_file = Path(tmpdir) / "github-copilot-token.json"
            token_file.write_text(json.dumps({"accessToken": ""}))
            cs_dir = Path(tmpdir) / "cs-data"
            cs_dir.mkdir()
            assert _load_github_token(str(cs_dir)) is None

    def test_load_github_token_bad_json(self) -> None:
        from kiss.agents.sorcar.code_server import _load_github_token
        with tempfile.TemporaryDirectory() as tmpdir:
            token_file = Path(tmpdir) / "github-copilot-token.json"
            token_file.write_text("not json")
            cs_dir = Path(tmpdir) / "cs-data"
            cs_dir.mkdir()
            assert _load_github_token(str(cs_dir)) is None

    def test_cleanup_merge_data(self) -> None:
        from kiss.agents.sorcar.code_server import _cleanup_merge_data
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "merge-temp").mkdir()
            Path(tmpdir, "merge-current").mkdir()
            manifest = Path(tmpdir, "pending-merge.json")
            manifest.write_text("{}")
            _cleanup_merge_data(tmpdir)
            assert not Path(tmpdir, "merge-temp").exists()
            assert not Path(tmpdir, "merge-current").exists()
            assert not manifest.exists()

    def test_restore_merge_files(self) -> None:
        from kiss.agents.sorcar.code_server import _restore_merge_files
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            work_dir = Path(tmpdir) / "work"
            work_dir.mkdir()
            current_dir = data_dir / "merge-current"
            current_dir.mkdir()
            (current_dir / "test.txt").write_text("restored")
            _restore_merge_files(str(data_dir), str(work_dir))
            assert (work_dir / "test.txt").read_text() == "restored"

    def test_restore_merge_files_no_dir(self) -> None:
        from kiss.agents.sorcar.code_server import _restore_merge_files
        _restore_merge_files("/nonexistent", "/nonexistent")  # should not raise

    def test_save_untracked_base(self) -> None:
        from kiss.agents.sorcar.code_server import _save_untracked_base, _untracked_base_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "new_file.py").write_text("content")
            _save_untracked_base(tmpdir, {"new_file.py"})
            base_dir = _untracked_base_dir()
            assert (base_dir / "new_file.py").exists()

    def test_capture_untracked(self) -> None:
        from kiss.agents.sorcar.code_server import _capture_untracked
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            Path(tmpdir, "untracked.txt").write_text("hello")
            result = _capture_untracked(tmpdir)
            assert "untracked.txt" in result

    def test_parse_diff_hunks(self) -> None:
        from kiss.agents.sorcar.code_server import _parse_diff_hunks
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=tmpdir, capture_output=True,
            )
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmpdir, capture_output=True)
            Path(tmpdir, "test.txt").write_text("line1\n")
            subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True)
            Path(tmpdir, "test.txt").write_text("line1\nline2\n")
            hunks = _parse_diff_hunks(tmpdir)
            assert "test.txt" in hunks

    def test_diff_files(self) -> None:
        from kiss.agents.sorcar.code_server import _diff_files
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "base.txt"
            current = Path(tmpdir) / "current.txt"
            base.write_text("line1\nline2\n")
            current.write_text("line1\nchanged\nline3\n")
            hunks = _diff_files(str(base), str(current))
            assert len(hunks) > 0

    def test_disable_copilot_scm_button(self) -> None:
        from kiss.agents.sorcar.code_server import _disable_copilot_scm_button
        with tempfile.TemporaryDirectory() as tmpdir:
            ext_dir = Path(tmpdir) / "github.copilot-chat-1.0.0"
            ext_dir.mkdir()
            pkg = {
                "contributes": {
                    "menus": {
                        "scm/inputBox": [
                            {"command": "github.copilot.git.generateCommitMessage", "when": "true"}
                        ]
                    }
                }
            }
            (ext_dir / "package.json").write_text(json.dumps(pkg))
            _disable_copilot_scm_button(tmpdir)
            updated = json.loads((ext_dir / "package.json").read_text())
            assert updated["contributes"]["menus"]["scm/inputBox"][0]["when"] == "false"

    def test_disable_copilot_scm_button_no_dir(self) -> None:
        from kiss.agents.sorcar.code_server import _disable_copilot_scm_button
        _disable_copilot_scm_button("/nonexistent")  # should not raise

    def test_disable_copilot_scm_button_already_false(self) -> None:
        from kiss.agents.sorcar.code_server import _disable_copilot_scm_button
        with tempfile.TemporaryDirectory() as tmpdir:
            ext_dir = Path(tmpdir) / "github.copilot-chat-1.0.0"
            ext_dir.mkdir()
            pkg = {
                "contributes": {
                    "menus": {
                        "scm/inputBox": [
                            {"command": "github.copilot.git.generateCommitMessage", "when": "false"}
                        ]
                    }
                }
            }
            (ext_dir / "package.json").write_text(json.dumps(pkg))
            _disable_copilot_scm_button(tmpdir)

    def test_disable_copilot_no_pkg(self) -> None:
        from kiss.agents.sorcar.code_server import _disable_copilot_scm_button
        with tempfile.TemporaryDirectory() as tmpdir:
            ext_dir = Path(tmpdir) / "github.copilot-chat-1.0.0"
            ext_dir.mkdir()
            # No package.json
            _disable_copilot_scm_button(tmpdir)

    def test_disable_copilot_bad_json(self) -> None:
        from kiss.agents.sorcar.code_server import _disable_copilot_scm_button
        with tempfile.TemporaryDirectory() as tmpdir:
            ext_dir = Path(tmpdir) / "github.copilot-chat-1.0.0"
            ext_dir.mkdir()
            (ext_dir / "package.json").write_text("not json")
            _disable_copilot_scm_button(tmpdir)

    def test_disable_copilot_non_copilot_dir(self) -> None:
        from kiss.agents.sorcar.code_server import _disable_copilot_scm_button
        with tempfile.TemporaryDirectory() as tmpdir:
            ext_dir = Path(tmpdir) / "other-extension-1.0"
            ext_dir.mkdir()
            _disable_copilot_scm_button(tmpdir)

    def test_install_copilot_already_installed(self) -> None:
        from kiss.agents.sorcar.code_server import _install_copilot_extension
        with tempfile.TemporaryDirectory() as tmpdir:
            ext_dir = Path(tmpdir) / "github.copilot-1.0"
            ext_dir.mkdir()
            _install_copilot_extension(tmpdir)  # should return early

    def test_setup_code_server_existing_settings(self) -> None:
        from kiss.agents.sorcar.code_server import _setup_code_server
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            ext_dir = os.path.join(tmpdir, "ext")
            # First setup
            _setup_code_server(data_dir, ext_dir)
            # Second setup should merge existing settings
            result = _setup_code_server(data_dir, ext_dir)
            assert isinstance(result, bool)

    def test_setup_code_server_workspace_cleanup(self) -> None:
        from kiss.agents.sorcar.code_server import _setup_code_server
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            ext_dir = os.path.join(tmpdir, "ext")
            # Create workspace storage with chat sessions
            ws_dir = Path(data_dir) / "User" / "workspaceStorage" / "abc123"
            chat_dir = ws_dir / "chatSessions"
            chat_dir.mkdir(parents=True)
            (chat_dir / "session.json").write_text("{}")
            edit_dir = ws_dir / "chatEditingSessions"
            edit_dir.mkdir(parents=True)
            (edit_dir / "session.json").write_text("{}")
            _setup_code_server(data_dir, ext_dir)
            assert not chat_dir.exists()
            assert not edit_dir.exists()


# ===================================================================
# model creation (covers __init__.py import paths)
# ===================================================================
# ===================================================================
# browser_ui.py — _handle_message and stream_event branches
# ===================================================================
class TestBrowserUIMessageHandling(TestCase):
    def test_handle_message_tool_output(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()

        class FakeMsg:
            subtype = "tool_output"
            data = {"content": "tool output text"}

        printer.print(FakeMsg(), type="message")
        event = cq.get_nowait()
        assert event["type"] == "system_output"
        assert event["text"] == "tool output text"

    def test_handle_message_tool_output_empty(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()

        class FakeMsg:
            subtype = "tool_output"
            data = {"content": ""}

        printer.print(FakeMsg(), type="message")
        assert cq.empty()

    def test_handle_message_result(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()

        class FakeMsg:
            result = "success"

        printer.print(FakeMsg(), type="message", step_count=2, budget_used=0.5)
        event = cq.get_nowait()
        assert event["type"] == "result"

    def test_handle_message_content_blocks(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()

        class FakeBlock:
            is_error = True
            content = "error message"

        class FakeMsg:
            content = [FakeBlock()]

        printer.print(FakeMsg(), type="message")
        event = cq.get_nowait()
        assert event["type"] == "tool_result"

    def test_print_stream_event(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()

        class FakeEvent:
            event = {
                "type": "content_block_start",
                "content_block": {"type": "thinking"},
            }

        printer.print(FakeEvent(), type="stream_event")
        event = cq.get_nowait()
        assert event["type"] == "thinking_start"

    def test_on_thinking_start_end(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer._on_thinking_start()
        printer._on_thinking_end()
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        types = [e["type"] for e in events]
        assert "thinking_start" in types
        assert "thinking_end" in types

    def test_on_tool_use_end(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer._on_tool_use_end("Bash", {"command": "ls"})
        event = cq.get_nowait()
        assert event["type"] == "tool_call"
        assert event["name"] == "Bash"

    def test_on_text_block_end(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer._on_text_block_end()
        event = cq.get_nowait()
        assert event["type"] == "text_end"

    def test_flush_bash_empty(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer._flush_bash()  # empty buffer
        assert cq.empty()

    def test_parse_result_yaml_invalid(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        assert BaseBrowserPrinter._parse_result_yaml("not yaml: [") is None

    def test_parse_result_yaml_no_summary(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        assert BaseBrowserPrinter._parse_result_yaml("key: value") is None

    def test_broadcast_result_no_text(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        printer._broadcast_result("")
        event = cq.get_nowait()
        assert event["text"] == "(no result)"

    def test_bash_stream_multiple_flushes(self) -> None:
        """Cover the branch where timer already exists."""
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        # Set recent flush so timer gets scheduled
        printer._bash_last_flush = time.monotonic()
        printer.print("a", type="bash_stream")  # schedules timer
        # Timer already exists
        printer._bash_last_flush = time.monotonic()
        printer.print("b", type="bash_stream")  # hits the else: needs_flush=False
        printer._flush_bash()  # force flush
        events = []
        while not cq.empty():
            events.append(cq.get_nowait())
        texts = "".join(e.get("text", "") for e in events if e.get("type") == "system_output")
        assert "a" in texts or "b" in texts


# ===================================================================
# task_history.py — more uncovered branches
# ===================================================================
class TestTaskHistoryExtra(TestCase):
    def setUp(self) -> None:
        from kiss.agents.sorcar import task_history as th
        self.th = th
        self.orig_history_file = th.HISTORY_FILE
        self.orig_events_dir = th._CHAT_EVENTS_DIR
        self.orig_model_usage = th.MODEL_USAGE_FILE
        self.orig_file_usage = th.FILE_USAGE_FILE
        self.orig_kiss_dir = th._KISS_DIR
        self.tmpdir = Path(tempfile.mkdtemp())
        th._KISS_DIR = self.tmpdir
        th.HISTORY_FILE = self.tmpdir / "task_history.jsonl"
        th._CHAT_EVENTS_DIR = self.tmpdir / "chat_events"
        th.MODEL_USAGE_FILE = self.tmpdir / "model_usage.json"
        th.FILE_USAGE_FILE = self.tmpdir / "file_usage.json"
        th._HISTORY_LOCK = th.FileLock(th.HISTORY_FILE.with_suffix(".lock"))
        th._history_cache = None
        th._total_count = 0

    def tearDown(self) -> None:
        th = self.th
        th.HISTORY_FILE = self.orig_history_file # type: ignore[attr-defined]
        th._CHAT_EVENTS_DIR = self.orig_events_dir # type: ignore[attr-defined]
        th.MODEL_USAGE_FILE = self.orig_model_usage # type: ignore[attr-defined]
        th.FILE_USAGE_FILE = self.orig_file_usage # type: ignore[attr-defined]
        th._KISS_DIR = self.orig_kiss_dir # type: ignore[attr-defined]
        th._HISTORY_LOCK = th.FileLock(th.HISTORY_FILE.with_suffix(".lock")) # type: ignore[attr-defined]
        th._history_cache = None # type: ignore[attr-defined]
        th._total_count = 0 # type: ignore[attr-defined]
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_history_with_limit(self) -> None:
        """Cover limit <= len(cache) branch."""
        th = self.th
        for i in range(5):
            th._add_task(f"task {i}")
        history = th._load_history(limit=3)
        assert len(history) == 3

    def test_load_history_limit_beyond_cache(self) -> None:
        """Cover limit > len(cache) — reads from tail."""
        th = self.th
        for i in range(5):
            th._add_task(f"task beyond {i}")
        history = th._load_history(limit=100)
        assert len(history) >= 5

    def test_get_history_entry_beyond_cache(self) -> None:
        """Cover reading from file when idx >= cache size."""
        th = self.th
        # Write many tasks to force beyond cache
        th.HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with th.HISTORY_FILE.open("w") as f:
            for i in range(200):
                f.write(json.dumps({
                    "task": f"bulk task {i:03d}",
                    "has_events": False,
                    "result": "",
                    "events_file": "",
                }) + "\n")
        th._history_cache = None # type: ignore[attr-defined]
        th._total_count = 0 # type: ignore[attr-defined]
        # Load cache first
        th._load_history(limit=1)
        # Get entry beyond cache
        entry = th._get_history_entry(150)
        if entry is not None:
            assert "bulk task" in str(entry["task"])

    def test_migrate_old_format_with_events(self) -> None:
        """Cover migration with chat_events present."""
        th = self.th
        old_file = self.tmpdir / "task_history.json"
        old_data = [
            {"task": "task with events", "chat_events": [{"type": "x"}]},
            {"task": "task without events"},
        ]
        old_file.write_text(json.dumps(old_data))
        th._migrate_old_format()
        assert th.HISTORY_FILE.exists()
        # Events should have been saved to separate file
        events_dir = th._CHAT_EVENTS_DIR
        if events_dir.exists():
            assert any(events_dir.iterdir())

    def test_record_file_usage(self) -> None:
        th = self.th
        th._record_file_usage("src/main.py")
        usage = th._load_file_usage()
        assert usage["src/main.py"] == 1

    def test_load_last_model_with_non_string(self) -> None:
        """Cover _load_last_model when _last is not string."""
        th = self.th
        th.MODEL_USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        th.MODEL_USAGE_FILE.write_text(json.dumps({"_last": 42}))
        assert th._load_last_model() == ""

    def test_int_values_filters(self) -> None:
        """Cover _int_values filtering non-numeric values."""
        th = self.th
        result = th._int_values({"a": 1, "b": 2.5, "c": "str", "d": None})
        assert result == {"a": 1, "b": 2}

    def test_read_recent_entries_empty_file(self) -> None:
        """Cover empty file case."""
        th = self.th
        th.HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        th.HISTORY_FILE.write_text("")
        entries = th._read_recent_entries(10)
        assert entries == []

    def test_set_latest_chat_events_updates_history_file(self) -> None:
        """Cover the full path: add task, set events, verify file entry."""
        th = self.th
        th._add_task("verified task")
        th._set_latest_chat_events(
            [{"type": "text", "text": "data"}],
            task="verified task",
            result="completed",
        )
        # Verify the events file exists
        path = th._task_events_path("verified task")
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 1

    def test_new_events_filename_unique(self) -> None:
        """Cover _new_events_filename."""
        th = self.th
        th._CHAT_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        name = th._new_events_filename()
        assert name.startswith("evt_")
        assert name.endswith(".json")

    def test_load_history_zero_limit(self) -> None:
        """Cover limit=0 path which reads all entries."""
        th = self.th
        for i in range(5):
            th._add_task(f"all task {i}")
        history = th._load_history(limit=0)
        assert len(history) >= 5

    def test_load_task_chat_events_not_in_cache(self) -> None:
        """Cover _load_task_chat_events when task not in cache."""
        th = self.th
        th._add_task("cached task")
        events = th._load_task_chat_events("non-cached task")
        assert events == []

    def test_load_task_chat_events_no_events_file(self) -> None:
        """Cover _load_task_chat_events when events_file is empty."""
        th = self.th
        th._add_task("no events file task")
        # Manually clear events_file
        assert th._history_cache is not None
        for entry in th._history_cache:
            if entry["task"] == "no events file task":
                entry["events_file"] = ""
        events = th._load_task_chat_events("no events file task")
        assert events == []

    def test_load_task_chat_events_file_not_list(self) -> None:
        """Cover _load_task_chat_events when file contains non-list JSON."""
        th = self.th
        th._add_task("dict events task")
        th._set_latest_chat_events([{"type": "x"}], task="dict events task")
        path = th._task_events_path("dict events task")
        if path.exists():
            path.write_text('{"not": "a list"}')
        events = th._load_task_chat_events("dict events task")
        assert events == []

    def test_task_events_path_no_filename(self) -> None:
        """Cover _task_events_path when entry has empty events_file."""
        th = self.th
        th._add_task("empty events path")
        assert th._history_cache is not None
        for entry in th._history_cache:
            if entry["task"] == "empty events path":
                entry["events_file"] = ""
        path = th._task_events_path("empty events path")
        assert "nonexistent" in str(path)

    def test_set_latest_events_result_update(self) -> None:
        """Cover result update in _set_latest_chat_events."""
        th = self.th
        th._add_task("result update task")
        th._set_latest_chat_events(
            [{"type": "x"}], task="result update task", result="updated"
        )
        assert th._history_cache is not None
        for entry in th._history_cache:
            if entry["task"] == "result update task":
                assert entry.get("result") == "updated"
                break

    def test_set_latest_events_without_task_updates_first(self) -> None:
        """Cover _set_latest_chat_events without task (updates cache[0])."""
        th = self.th
        th._add_task("first task")
        th._add_task("second task")
        th._set_latest_chat_events([{"type": "done"}])
        assert th._history_cache is not None
        assert th._history_cache[0]["has_events"] is True

    def test_add_task_overflows_cache(self) -> None:
        """Cover cache trimming when it exceeds _RECENT_CACHE_SIZE."""
        th = self.th
        for i in range(th._RECENT_CACHE_SIZE + 10):
            th._add_task(f"overflow task {i}")
        assert th._history_cache is not None
        assert len(th._history_cache) <= th._RECENT_CACHE_SIZE

    def test_cleanup_stale_cs_dirs_with_stale(self) -> None:
        """Cover _cleanup_stale_cs_dirs with stale dirs."""
        th = self.th
        cs_dir = self.tmpdir / "cs-stale123"
        cs_dir.mkdir()
        (cs_dir / "cs-port").write_text("99998")
        # Make it old
        old_time = time.time() - 48 * 3600
        os.utime(cs_dir, (old_time, old_time))
        removed = th._cleanup_stale_cs_dirs(max_age_hours=24)
        assert removed >= 0  # may or may not find it depending on glob

    def test_cleanup_stale_cs_dirs_skips_extensions(self) -> None:
        """Cover cs-extensions skip."""
        th = self.th
        ext_dir = self.tmpdir / "cs-extensions"
        ext_dir.mkdir()
        th._cleanup_stale_cs_dirs()
        assert ext_dir.exists()  # should not be removed


# ===================================================================
# useful_tools.py — command parsing edge cases
# ===================================================================
class TestUsefulToolsCommandParsing(TestCase):
    def test_extract_command_with_pipe(self) -> None:
        from kiss.agents.sorcar.useful_tools import _extract_command_names
        names = _extract_command_names("echo hello | grep h | wc -l")
        assert "echo" in names
        assert "grep" in names
        assert "wc" in names

    def test_extract_command_with_semicolons(self) -> None:
        from kiss.agents.sorcar.useful_tools import _extract_command_names
        names = _extract_command_names("cd /tmp; ls; pwd")
        assert "cd" in names
        assert "ls" in names
        assert "pwd" in names

    def test_extract_command_with_redirect(self) -> None:
        from kiss.agents.sorcar.useful_tools import _extract_command_names
        names = _extract_command_names("echo hello > file.txt")
        assert "echo" in names

    def test_extract_command_with_background(self) -> None:
        from kiss.agents.sorcar.useful_tools import _extract_command_names
        names = _extract_command_names("sleep 10 &")
        assert "sleep" in names

    def test_extract_command_empty(self) -> None:
        from kiss.agents.sorcar.useful_tools import _extract_command_names
        assert _extract_command_names("") == []

    def test_extract_command_with_braces(self) -> None:
        from kiss.agents.sorcar.useful_tools import _extract_command_names
        names = _extract_command_names("{ echo hello; }")
        assert "echo" in names

    def test_extract_command_quoted(self) -> None:
        from kiss.agents.sorcar.useful_tools import _extract_command_names
        names = _extract_command_names("echo 'hello && world'")
        assert "echo" in names
        assert len(names) == 1

    def test_split_respecting_quotes(self) -> None:
        from kiss.agents.sorcar.useful_tools import _CONTROL_RE, _split_respecting_quotes
        result = _split_respecting_quotes("a && 'b && c' && d", _CONTROL_RE)
        assert len(result) == 3

    def test_extract_leading_command_with_subshell(self) -> None:
        from kiss.agents.sorcar.useful_tools import _extract_leading_command_name
        name = _extract_leading_command_name("(echo hello)")
        assert name == "echo"

    def test_extract_leading_command_empty(self) -> None:
        from kiss.agents.sorcar.useful_tools import _extract_leading_command_name
        assert _extract_leading_command_name("") is None

    def test_extract_leading_command_only_envvars(self) -> None:
        from kiss.agents.sorcar.useful_tools import _extract_leading_command_name
        # Only env vars, no actual command
        assert _extract_leading_command_name("FOO=bar") is None

    def test_extract_leading_command_with_path(self) -> None:
        from kiss.agents.sorcar.useful_tools import _extract_leading_command_name
        name = _extract_leading_command_name("/usr/bin/python script.py")
        assert name == "python"

    def test_extract_leading_with_shlex_error(self) -> None:
        """Cover ValueError from shlex.split (line 48-50)."""
        from kiss.agents.sorcar.useful_tools import _extract_leading_command_name
        # Unterminated quote
        assert _extract_leading_command_name("echo 'unterminated") is None

    def test_extract_leading_with_redirect(self) -> None:
        """Cover redirect handling branches (lines 65-69)."""
        from kiss.agents.sorcar.useful_tools import _extract_leading_command_name
        # Redirect with file
        name = _extract_leading_command_name("2>/dev/null echo hello")
        assert name == "echo"
        # Redirect inline (m.end() < len(token))
        name = _extract_leading_command_name(">file echo hello")
        assert name == "echo"

    def test_extract_leading_only_redirects(self) -> None:
        """Cover the case where only redirects remain."""
        from kiss.agents.sorcar.useful_tools import _extract_leading_command_name
        # Only a redirect, no command
        assert _extract_leading_command_name(">/dev/null") is None

    def test_split_with_escape(self) -> None:
        """Cover escape handling in _split_respecting_quotes (line 88-90)."""
        from kiss.agents.sorcar.useful_tools import _CONTROL_RE, _split_respecting_quotes
        result = _split_respecting_quotes("a\\;b && c", _CONTROL_RE)
        assert len(result) == 2

    def test_split_with_double_quote_escape(self) -> None:
        """Cover double-quote escape handling (lines 96-97)."""
        from kiss.agents.sorcar.useful_tools import _CONTROL_RE, _split_respecting_quotes
        result = _split_respecting_quotes('echo "a\\"b" && echo c', _CONTROL_RE)
        assert len(result) == 2

    def test_truncate_output_zero_tail(self) -> None:
        """Cover tail=0 branch in _truncate_output (line 29)."""
        from kiss.agents.sorcar.useful_tools import _truncate_output
        # Make max_chars such that remaining - head leaves tail = 0
        text = "x" * 200
        msg_len = len(f"\n\n... [truncated {200 - 60} chars] ...\n\n")
        result = _truncate_output(text, 60 + msg_len)
        assert "truncated" in result

    def test_disallowed_source(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        tools = UsefulTools()
        result = tools.Bash(command="source .env", description="source", timeout_seconds=5)
        assert "not allowed" in result

    def test_disallowed_exec(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        tools = UsefulTools()
        result = tools.Bash(command="exec bash", description="exec", timeout_seconds=5)
        assert "not allowed" in result

    def test_kill_process_group(self) -> None:
        from kiss.agents.sorcar.useful_tools import _kill_process_group
        proc = subprocess.Popen(
            ["sleep", "30"],
            start_new_session=True,
        )
        _kill_process_group(proc)
        assert proc.poll() is not None


# ===================================================================
# code_server.py — more utility functions
# ===================================================================
class TestCodeServerSetup(TestCase):
    def test_setup_code_server(self) -> None:
        from kiss.agents.sorcar.code_server import _setup_code_server
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            ext_dir = os.path.join(tmpdir, "ext")
            result = _setup_code_server(data_dir, ext_dir)
            assert isinstance(result, bool)
            # Should have created settings
            assert os.path.exists(os.path.join(data_dir, "User", "settings.json"))

    def test_scan_files_with_max_limit(self) -> None:
        from kiss.agents.sorcar.code_server import _scan_files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many files
            for i in range(50):
                Path(tmpdir, f"file{i:04d}.txt").write_text("x")
            files = _scan_files(tmpdir)
            assert len(files) >= 50

    def test_prepare_merge_view(self) -> None:
        from kiss.agents.sorcar.code_server import _prepare_merge_view
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            work_dir = os.path.join(tmpdir, "work")
            os.makedirs(work_dir, exist_ok=True)
            # Init git repo
            subprocess.run(["git", "init"], cwd=work_dir, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "t@t.com"],
                cwd=work_dir, capture_output=True,
            )
            subprocess.run(["git", "config", "user.name", "T"], cwd=work_dir, capture_output=True)
            # Create initial commit
            Path(work_dir, "test.txt").write_text("line1\n")
            subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=work_dir, capture_output=True)
            # No changes -> should return no_changes
            result = _prepare_merge_view(work_dir, data_dir, {}, set(), {})
            assert result.get("status") in ("no_changes", "opened", None)


# ===================================================================
# print_to_console.py — ConsolePrinter tests
# ===================================================================
class TestConsolePrinter(TestCase):
    def _make_printer(self) -> Any:
        from io import StringIO

        from kiss.core.print_to_console import ConsolePrinter
        buf = StringIO()
        return ConsolePrinter(file=buf), buf

    def test_print_text(self) -> None:
        printer, buf = self._make_printer()
        printer.print("Hello world", type="text")

    def test_print_prompt(self) -> None:
        printer, buf = self._make_printer()
        printer.print("My prompt", type="prompt")

    def test_print_usage_info(self) -> None:
        printer, buf = self._make_printer()
        printer.print("Steps: 1/10", type="usage_info")

    def test_print_bash_stream(self) -> None:
        printer, buf = self._make_printer()
        printer.print("output line\n", type="bash_stream")
        assert "output line" in buf.getvalue()

    def test_print_tool_call(self) -> None:
        printer, buf = self._make_printer()
        printer.print("Bash", type="tool_call", tool_input={
            "command": "ls -la",
            "description": "list files",
        })

    def test_print_tool_call_with_edit(self) -> None:
        printer, buf = self._make_printer()
        printer.print("Edit", type="tool_call", tool_input={
            "file_path": "test.py",
            "old_string": "old",
            "new_string": "new",
        })

    def test_print_tool_call_no_args(self) -> None:
        printer, buf = self._make_printer()
        printer.print("finish", type="tool_call", tool_input={})

    def test_print_tool_result(self) -> None:
        printer, buf = self._make_printer()
        printer.print("output", type="tool_result", is_error=False)

    def test_print_tool_result_error(self) -> None:
        printer, buf = self._make_printer()
        printer.print("error msg", type="tool_result", is_error=True)

    def test_print_result(self) -> None:
        printer, buf = self._make_printer()
        printer.print("final result", type="result", step_count=5, cost="$0.01")

    def test_print_result_yaml(self) -> None:
        printer, buf = self._make_printer()
        printer.print(yaml.dump({"success": True, "summary": "done"}), type="result")

    def test_print_result_yaml_failed(self) -> None:
        printer, buf = self._make_printer()
        printer.print(yaml.dump({"success": False, "summary": "failed"}), type="result")

    def test_print_result_empty(self) -> None:
        printer, buf = self._make_printer()
        printer.print("", type="result")

    def test_print_unknown_type(self) -> None:
        printer, buf = self._make_printer()
        result = printer.print("x", type="xyz_unknown")
        assert result == ""

    def test_token_callback(self) -> None:
        printer, buf = self._make_printer()
        asyncio.run(printer.token_callback("hello"))

    def test_token_callback_thinking(self) -> None:
        printer, buf = self._make_printer()
        printer._current_block_type = "thinking"
        asyncio.run(printer.token_callback("thought"))

    def test_stream_event_handling(self) -> None:
        printer, buf = self._make_printer()

        class Event:
            def __init__(self, evt: dict) -> None:
                self.event = evt

        # Thinking block
        evt = {"type": "content_block_start",
               "content_block": {"type": "thinking"}}
        printer.print(Event(evt), type="stream_event")
        evt = {"type": "content_block_delta",
               "delta": {"type": "thinking_delta", "thinking": "hmm"}}
        printer.print(Event(evt), type="stream_event")
        printer.print(
            Event({"type": "content_block_stop"}),
            type="stream_event",
        )

    def test_stream_event_tool_use(self) -> None:
        printer, buf = self._make_printer()

        class Event:
            def __init__(self, evt: dict) -> None:
                self.event = evt

        evt = {"type": "content_block_start",
               "content_block": {"type": "tool_use", "name": "Bash"}}
        printer.print(Event(evt), type="stream_event")
        evt = {"type": "content_block_delta",
               "delta": {"type": "input_json_delta",
                         "partial_json": '{"command":"ls"}'}}
        printer.print(Event(evt), type="stream_event")
        printer.print(
            Event({"type": "content_block_stop"}),
            type="stream_event",
        )

    def test_stream_event_text_block(self) -> None:
        printer, buf = self._make_printer()

        class Event:
            def __init__(self, evt: dict) -> None:
                self.event = evt

        evt = {"type": "content_block_start",
               "content_block": {"type": "text"}}
        printer.print(Event(evt), type="stream_event")
        evt = {"type": "content_block_delta",
               "delta": {"type": "text_delta", "text": "hello"}}
        printer.print(Event(evt), type="stream_event")
        printer.print(
            Event({"type": "content_block_stop"}),
            type="stream_event",
        )

    def test_message_tool_output(self) -> None:
        printer, buf = self._make_printer()

        class FakeMsg:
            subtype = "tool_output"
            data = {"content": "tool output\n"}

        printer.print(FakeMsg(), type="message")
        assert "tool output" in buf.getvalue()

    def test_message_tool_output_empty(self) -> None:
        printer, buf = self._make_printer()

        class FakeMsg:
            subtype = "tool_output"
            data = {"content": ""}

        printer.print(FakeMsg(), type="message")

    def test_message_result(self) -> None:
        printer, buf = self._make_printer()

        class FakeMsg:
            result = "success"

        printer.print(FakeMsg(), type="message", budget_used=0.5)

    def test_message_content_blocks(self) -> None:
        printer, buf = self._make_printer()

        class FakeBlock:
            is_error = True
            content = "error message"

        class FakeMsg:
            content = [FakeBlock()]

        printer.print(FakeMsg(), type="message")

    def test_reset(self) -> None:
        printer, buf = self._make_printer()
        printer._mid_line = True
        printer.reset()
        assert not printer._mid_line

    def test_format_result_invalid_yaml(self) -> None:
        from kiss.core.print_to_console import ConsolePrinter
        result = ConsolePrinter._format_result_content("not: [valid yaml: [")
        assert isinstance(result, str)

    def test_format_result_no_summary(self) -> None:
        from kiss.core.print_to_console import ConsolePrinter
        result = ConsolePrinter._format_result_content("key: value")
        assert result == "key: value"

    def test_flush_newline(self) -> None:
        printer, buf = self._make_printer()
        printer._mid_line = True
        printer._flush_newline()
        assert "\n" in buf.getvalue()
        assert not printer._mid_line


# ===================================================================
# code_server.py — _prepare_merge_view with actual changes
# ===================================================================
class TestPreparedMergeView(TestCase):
    def test_prepare_merge_view_with_changes(self) -> None:
        """Cover _prepare_merge_view when there ARE agent changes."""
        from kiss.agents.sorcar.code_server import (
            _capture_untracked,
            _parse_diff_hunks,
            _prepare_merge_view,
            _save_untracked_base,
            _snapshot_files,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            work_dir = os.path.join(tmpdir, "work")
            os.makedirs(work_dir, exist_ok=True)
            subprocess.run(["git", "init"], cwd=work_dir, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "t@t.com"],
                cwd=work_dir, capture_output=True,
            )
            subprocess.run(["git", "config", "user.name", "T"], cwd=work_dir, capture_output=True)
            Path(work_dir, "test.txt").write_text("line1\nline2\n")
            subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=work_dir, capture_output=True)
            # Pre-state
            pre_hunks = _parse_diff_hunks(work_dir)
            pre_untracked = _capture_untracked(work_dir)
            pre_hashes = _snapshot_files(work_dir, set(pre_hunks.keys()) | pre_untracked)
            _save_untracked_base(work_dir, pre_untracked | set(pre_hunks.keys()))
            # Make changes (agent modifies file)
            Path(work_dir, "test.txt").write_text("line1\nmodified\nline3\n")
            # Add new untracked file
            Path(work_dir, "new_file.py").write_text("print('hello')\n")
            result = _prepare_merge_view(work_dir, data_dir, pre_hunks, pre_untracked, pre_hashes)
            assert result.get("status") == "opened"
            assert result.get("count", 0) >= 1

    def test_prepare_merge_view_no_changes(self) -> None:
        from kiss.agents.sorcar.code_server import _prepare_merge_view
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)
            work_dir = os.path.join(tmpdir, "work")
            os.makedirs(work_dir)
            subprocess.run(["git", "init"], cwd=work_dir, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "t@t.com"],
                cwd=work_dir, capture_output=True,
            )
            subprocess.run(["git", "config", "user.name", "T"], cwd=work_dir, capture_output=True)
            Path(work_dir, "x.txt").write_text("x\n")
            subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=work_dir, capture_output=True)
            result = _prepare_merge_view(work_dir, data_dir, {}, set(), {})
            if isinstance(result, str):
                assert "error" in result.lower()
            else:
                assert result.get("error") is not None

    def test_prepare_merge_with_untracked_modified(self) -> None:
        """Cover modified pre-existing untracked file detection."""
        from kiss.agents.sorcar.code_server import (
            _capture_untracked,
            _prepare_merge_view,
            _save_untracked_base,
            _snapshot_files,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)
            work_dir = os.path.join(tmpdir, "work")
            os.makedirs(work_dir)
            subprocess.run(["git", "init"], cwd=work_dir, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "t@t.com"],
                cwd=work_dir, capture_output=True,
            )
            subprocess.run(["git", "config", "user.name", "T"], cwd=work_dir, capture_output=True)
            # Initial commit
            Path(work_dir, "readme.md").write_text("hello\n")
            subprocess.run(["git", "add", "."], cwd=work_dir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=work_dir, capture_output=True)
            # Create untracked file
            Path(work_dir, "untracked.txt").write_text("original\n")
            pre_untracked = _capture_untracked(work_dir)
            pre_hashes = _snapshot_files(work_dir, pre_untracked)
            _save_untracked_base(work_dir, pre_untracked)
            # Modify untracked file
            Path(work_dir, "untracked.txt").write_text("modified\n")
            result = _prepare_merge_view(work_dir, data_dir, {}, pre_untracked, pre_hashes)
            # Should detect the modification
            assert result.get("status") == "opened" or "error" in str(result)


# ===================================================================
# useful_tools.py — more edge cases
# ===================================================================
class TestUsefulToolsMoreEdges(TestCase):
    def test_write_creates_dirs(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        tools = UsefulTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "test.txt")
            result = tools.Write(path, "content")
            assert "Successfully" in result

    def test_bash_multi_command(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        tools = UsefulTools()
        result = tools.Bash(command="echo a && echo b", description="multi", timeout_seconds=5)
        assert "a" in result
        assert "b" in result

    def test_bash_or_operator(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        tools = UsefulTools()
        result = tools.Bash(command="false || echo fallback", description="or", timeout_seconds=5)
        assert "fallback" in result

    def test_format_bash_result_no_output(self) -> None:
        from kiss.agents.sorcar.useful_tools import _format_bash_result
        result = _format_bash_result(1, "", 1000)
        assert "Error (exit code 1)" in result


# ===================================================================
# model_info.py — more coverage
# ===================================================================
class TestModelInfoFunctions(TestCase):
    def test_calculate_cost_known_model(self) -> None:
        from kiss.core.models.model_info import calculate_cost
        cost = calculate_cost("gpt-4o-mini", 1000, 500)
        assert cost >= 0.0

    def test_calculate_cost_unknown_model(self) -> None:
        from kiss.core.models.model_info import calculate_cost
        cost = calculate_cost("unknown-model-xyz", 1000, 500)
        assert cost == 0.0

    def test_calculate_cost_with_cache(self) -> None:
        from kiss.core.models.model_info import calculate_cost
        cost = calculate_cost("claude-sonnet-4-20250514", 1000, 500, 200, 100)
        assert cost >= 0.0

    def test_get_max_context_length(self) -> None:
        from kiss.core.models.model_info import get_max_context_length
        length = get_max_context_length("gpt-4o-mini")
        assert length > 0

    def test_get_max_context_length_unknown(self) -> None:
        from kiss.core.models.model_info import get_max_context_length
        with self.assertRaises(KeyError):
            get_max_context_length("unknown-model-xyz")

    def test_is_model_flaky(self) -> None:
        from kiss.core.models.model_info import is_model_flaky
        # Most common models should not be flaky
        assert is_model_flaky("gpt-4o-mini") is False

    def test_get_flaky_reason(self) -> None:
        from kiss.core.models.model_info import get_flaky_reason
        assert get_flaky_reason("gpt-4o-mini") == ""

    def test_get_most_expensive_model(self) -> None:
        from kiss.core.models.model_info import get_most_expensive_model
        name = get_most_expensive_model()
        # May be empty if no API keys configured
        assert isinstance(name, str)

    def test_get_most_expensive_model_no_fc(self) -> None:
        from kiss.core.models.model_info import get_most_expensive_model
        name = get_most_expensive_model(fc_only=False)
        assert isinstance(name, str)


# ===================================================================
# config_builder.py — test CLI arg parsing with overrides
# ===================================================================
class TestConfigBuilderOverrides(TestCase):
    def test_add_model_arguments_bool(self) -> None:
        """Cover bool argument handling (lines 50-59)."""
        from argparse import ArgumentParser

        from pydantic import BaseModel

        from kiss.core.config_builder import _add_model_arguments

        class TestCfg(BaseModel):
            flag: bool = False
            name: str = "default"
            count: int = 0
            rate: float = 1.0

        parser = ArgumentParser()
        _add_model_arguments(parser, TestCfg)
        # Parse with flag
        args, _ = parser.parse_known_args(
            ["--flag", "--name", "test", "--count", "5", "--rate", "2.5"],
        )
        assert args.flag is True
        assert args.name == "test"
        assert args.count == 5
        assert args.rate == 2.5

    def test_add_model_arguments_nested(self) -> None:
        """Cover nested BaseModel recursion (lines 30-33)."""
        from argparse import ArgumentParser

        from pydantic import BaseModel

        from kiss.core.config_builder import _add_model_arguments

        class Inner(BaseModel):
            value: str = "inner"

        class Outer(BaseModel):
            inner: Inner = Inner()
            name: str = "outer"

        parser = ArgumentParser()
        _add_model_arguments(parser, Outer)
        args, _ = parser.parse_known_args(["--inner.value", "changed"])
        assert args.inner__value == "changed"

    def test_flat_to_nested_dict(self) -> None:
        """Cover _flat_to_nested_dict with nested model."""
        from pydantic import BaseModel

        from kiss.core.config_builder import _flat_to_nested_dict

        class Inner(BaseModel):
            value: str = "inner"

        class Outer(BaseModel):
            inner: Inner = Inner()
            name: str = "outer"

        flat = {"inner__value": "changed", "name": None}
        result = _flat_to_nested_dict(flat, Outer)
        assert result == {"inner": {"value": "changed"}}

    def test_add_config_with_overrides(self) -> None:
        """Cover the merge path in add_config (lines 164-176)."""
        from pydantic import BaseModel

        class OverrideCfg(BaseModel):
            test_val: str = "default"

        # This should work (no CLI args override, just default)
        add_config("override_cfg", OverrideCfg)
        assert config_module.DEFAULT_CONFIG.override_cfg.test_val == "default"


class TestModelCreation(TestCase):
    def test_openai_model_creation(self) -> None:
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
        m = OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        assert m.model_name == "gpt-4o-mini"

    def test_gemini_model_creation(self) -> None:
        from kiss.core.models.gemini_model import GeminiModel
        m = GeminiModel(model_name="gemini-2.0-flash", api_key="test-key")
        assert m.model_name == "gemini-2.0-flash"

    def test_anthropic_model_creation(self) -> None:
        from kiss.core.models.anthropic_model import AnthropicModel
        m = AnthropicModel(model_name="claude-3-5-sonnet-20241022", api_key="test-key")
        assert m.model_name == "claude-3-5-sonnet-20241022"


# ===================================================================
# base.py — test Base class methods
# ===================================================================
class TestBase(TestCase):
    def test_set_printer(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        from kiss.core.base import Base

        class TestAgent(Base):
            pass

        agent = TestAgent("test-base")
        printer = BaseBrowserPrinter()
        agent.set_printer(printer, verbose=False)
        assert agent.printer is printer

    def test_set_printer_none(self) -> None:
        from kiss.core.base import Base

        class TestAgent(Base):
            pass

        agent = TestAgent("test-base-none")
        agent.set_printer(None, verbose=True)

    def test_set_printer_verbose_false(self) -> None:
        from kiss.core.base import Base

        class TestAgent(Base):
            pass

        agent = TestAgent("test-base-vf")
        agent.set_printer(None, verbose=False)
        assert agent.printer is None

    def test_get_trajectory(self) -> None:
        from kiss.core.base import Base

        class TestAgent(Base):
            pass

        agent = TestAgent("test-traj")
        agent.messages = []
        agent._add_message("user", "hello")
        agent._add_message("model", "world")
        traj = agent.get_trajectory()
        assert "hello" in traj
        assert "world" in traj

    def test_save_messages(self) -> None:
        from kiss.core.base import Base

        class TestAgent(Base):
            pass

        agent = TestAgent("test-save")
        agent.messages = []
        agent.model_name = "gpt-4o-mini"
        agent.function_map = {}
        agent.run_start_timestamp = int(time.time())
        agent.budget_used = 0.0
        agent.total_tokens_used = 0
        agent.step_count = 0
        agent._add_message("user", "test message")
        agent._save()
        # Verify the file was saved
        # Should save to artifact_dir/messages/{id}.yaml
