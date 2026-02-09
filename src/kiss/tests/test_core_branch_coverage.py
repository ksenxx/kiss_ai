"""Test suite for increasing branch coverage of KISS core components.

These tests target specific branches and edge cases in:
- base.py: Base class for agents
- utils.py: Utility functions
- model_info.py: Model information and lookup
- simple_formatter.py and compact_formatter.py: Formatter implementations
- config.py: Configuration classes
"""

import os
from pathlib import Path

import pytest

from kiss.core import config as config_module
from kiss.core.base import CODING_INSTRUCTIONS, Base
from kiss.core.compact_formatter import CompactFormatter
from kiss.core.config import (
    AgentConfig,
    APIKeysConfig,
    Config,
    DockerConfig,
    KISSCodingAgentConfig,
    RelentlessCodingAgentConfig,
)
from kiss.core.simple_formatter import SimpleFormatter, _left_aligned_heading
from kiss.core.utils import (
    get_config_value,
    is_subpath,
    read_project_file,
    read_project_file_from_package,
    resolve_path,
)

FORMATTER_CLASSES = [SimpleFormatter, CompactFormatter]


@pytest.fixture
def temp_dir(tmp_path):
    original = os.getcwd()
    resolved = tmp_path.resolve()
    os.chdir(resolved)
    yield resolved
    os.chdir(original)


@pytest.fixture
def verbose_config():
    original = config_module.DEFAULT_CONFIG.agent.verbose
    yield
    config_module.DEFAULT_CONFIG.agent.verbose = original


@pytest.fixture
def base_state():
    original_counter = Base.agent_counter
    original_budget = Base.global_budget_used
    yield
    Base.agent_counter = original_counter
    Base.global_budget_used = original_budget


class TestBaseClass:
    def test_basic_init_and_counter(self, base_state):
        initial = Base.agent_counter
        agent = Base("test_agent")
        assert agent.name == "test_agent"
        assert agent.base_dir == ""
        assert isinstance(agent.id, int)
        Base("agent2")
        assert Base.agent_counter == initial + 2

    def test_run_state_and_messages(self, base_state):
        Base.global_budget_used = 5.5
        agent = Base("test")
        agent._init_run_state("gpt-4o", ["func1"])
        assert agent.model_name == "gpt-4o"
        assert agent.function_map == ["func1"]
        assert agent.messages == []
        assert isinstance(agent.run_start_timestamp, int)

        agent._add_message("user", "Hello")
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "user"
        assert agent.messages[0]["content"] == "Hello"

        state = agent._build_state_dict()
        assert state["model"] == "gpt-4o"
        assert state["global_budget_used"] == 5.5

        trajectory = agent.get_trajectory()
        assert isinstance(trajectory, str)
        assert "Hello" in trajectory

    def test_build_state_dict_unknown_model(self, base_state):
        agent = Base("test")
        agent._init_run_state("unknown-model-xyz", [])
        state = agent._build_state_dict()
        assert state["max_tokens"] is None

    def test_coding_instructions_constant(self):
        assert CODING_INSTRUCTIONS is not None
        assert len(CODING_INSTRUCTIONS) > 0

    def test_save_creates_trajectory_file(self, base_state, tmp_path):
        original_artifact_dir = config_module.DEFAULT_CONFIG.agent.artifact_dir
        config_module.DEFAULT_CONFIG.agent.artifact_dir = str(tmp_path)
        try:
            agent = Base("test_save_agent")
            agent._init_run_state("gpt-4.1-mini", [])
            agent._add_message("user", "Test message")
            agent._save()
            trajectories_dir = tmp_path / "trajectories"
            assert trajectories_dir.exists()
            files = list(trajectories_dir.glob("trajectory_test_save_agent_*.yaml"))
            assert len(files) == 1
        finally:
            config_module.DEFAULT_CONFIG.agent.artifact_dir = original_artifact_dir


class TestUtils:
    def test_resolve_path(self, temp_dir):
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")
        assert resolve_path(str(test_file), str(temp_dir)) == test_file.resolve()

    def test_resolve_path_absolute(self, tmp_path):
        abs_path = str(tmp_path / "test.txt")
        assert resolve_path(abs_path, "/some/other/base") == Path(abs_path).resolve()

    def test_is_subpath(self, temp_dir):
        parent = temp_dir / "parent"
        child = parent / "child"
        parent.mkdir()
        assert is_subpath(child, [parent])
        assert not is_subpath(Path("/etc/passwd"), [temp_dir])

    def test_get_config_value(self):
        result = get_config_value(None, config_module.DEFAULT_CONFIG.agent, "verbose")
        assert isinstance(result, bool)

    def test_get_config_value_prefers_explicit(self):
        result = get_config_value("explicit", config_module.DEFAULT_CONFIG.agent, "verbose")
        assert result == "explicit"

    def test_get_config_value_uses_default(self):
        class ConfigWithNone:
            nonexistent = None

        result = get_config_value(None, ConfigWithNone(), "nonexistent", default="fallback")
        assert result == "fallback"

    def test_get_config_value_raises_on_missing(self):
        class EmptyConfig:
            pass

        with pytest.raises(ValueError, match="No value provided"):
            get_config_value(None, EmptyConfig(), "nonexistent_attr")

    def test_read_project_file_existing(self):
        try:
            content = read_project_file("src/kiss/core/__init__.py")
            assert isinstance(content, str)
            assert len(content) > 0
        except Exception:
            pytest.skip("Project file not accessible in test environment")

    def test_read_project_file_not_found(self):
        from kiss.core.kiss_error import KISSError

        with pytest.raises(KISSError, match="Could not find"):
            read_project_file("nonexistent/path/to/file.txt")

    def test_read_project_file_from_package_not_found(self):
        from kiss.core.kiss_error import KISSError

        with pytest.raises(KISSError, match="Could not find"):
            read_project_file_from_package("nonexistent_file.txt")


class TestFormatters:
    @pytest.mark.parametrize("formatter_class", FORMATTER_CLASSES)
    @pytest.mark.parametrize("verbose", [True, False])
    def test_format_methods(self, verbose_config, formatter_class, verbose):
        config_module.DEFAULT_CONFIG.agent.verbose = verbose
        formatter = formatter_class()

        result = formatter.format_message({"role": "user", "content": "Hello"})
        if verbose:
            assert "user" in result.lower() or "Hello" in result
        else:
            assert result == ""

        messages = [{"role": "user", "content": "Hello"}, {"role": "model", "content": "Hi"}]
        result = formatter.format_messages(messages)
        if verbose:
            assert "Hello" in result
        else:
            assert result == ""

    @pytest.mark.parametrize("formatter_class", FORMATTER_CLASSES)
    @pytest.mark.parametrize("verbose", [True, False])
    def test_all_print_methods(self, verbose_config, formatter_class, verbose):
        config_module.DEFAULT_CONFIG.agent.verbose = verbose
        formatter = formatter_class()
        formatter.print_message({"role": "user", "content": "Test"})
        formatter.print_messages([{"role": "user", "content": "Test"}])
        formatter.print_status("Status")
        formatter.print_error("Error")
        formatter.print_warning("Warning")
        formatter.print_label_and_value("Label", "Value")

    @pytest.mark.parametrize("formatter_class", FORMATTER_CLASSES)
    def test_print_methods_no_console(self, verbose_config, formatter_class, capsys):
        config_module.DEFAULT_CONFIG.agent.verbose = True
        formatter = formatter_class()
        formatter._console = None
        formatter._stderr_console = None
        formatter.print_message({"role": "user", "content": "Test content"})
        formatter.print_messages([{"role": "user", "content": "Msg1"}])
        formatter.print_status("Status")
        formatter.print_error("Error")
        formatter.print_warning("Warning")
        formatter.print_label_and_value("Label", "Value")
        captured = capsys.readouterr()
        assert "Status" in captured.out
        assert "Error" in captured.err
        assert "Warning" in captured.out

    @pytest.mark.parametrize("formatter_class", FORMATTER_CLASSES)
    def test_print_methods_with_console(self, verbose_config, formatter_class):
        from io import StringIO

        from rich.console import Console

        config_module.DEFAULT_CONFIG.agent.verbose = True
        formatter = formatter_class()
        output = StringIO()
        err_output = StringIO()
        formatter._console = Console(file=output, force_terminal=True)
        formatter._stderr_console = Console(file=err_output, force_terminal=True)
        formatter.print_status("Status")
        formatter.print_error("Error")
        formatter.print_warning("Warning")
        formatter.print_label_and_value("Label", "Value")


class TestSimpleFormatterSpecific:
    def test_format_message_missing_keys(self, verbose_config):
        config_module.DEFAULT_CONFIG.agent.verbose = True
        formatter = SimpleFormatter()
        result = formatter.format_message({})
        assert 'role=""' in result


class TestCompactFormatterSpecific:
    def test_truncates_long_content(self, verbose_config):
        config_module.DEFAULT_CONFIG.agent.verbose = True
        formatter = CompactFormatter()
        message = {"role": "user", "content": "A" * 200}
        result = formatter.format_message(message)
        assert len(result) < 200

    def test_replaces_newlines(self, verbose_config):
        config_module.DEFAULT_CONFIG.agent.verbose = True
        formatter = CompactFormatter()
        message = {"role": "user", "content": "line1\nline2"}
        result = formatter.format_message(message)
        assert "line1" in result and "line2" in result
        assert "line1\nline2" not in result

    def test_unknown_role(self, verbose_config):
        config_module.DEFAULT_CONFIG.agent.verbose = True
        formatter = CompactFormatter()
        result = formatter.format_message({"content": "Hello"})
        assert "[unknown]" in result


class TestLeftAlignedHeading:
    @pytest.mark.parametrize("tag,expected_count", [("h1", 1), ("h2", 2), ("h3", 1)])
    def test_heading_tags(self, tag, expected_count):
        from rich.text import Text

        class MockHeading:
            def __init__(self, t):
                self.tag = t
                self.text = Text("Heading")

        results = list(_left_aligned_heading(MockHeading(tag), None, None))
        assert len(results) == expected_count


class TestConfigClasses:
    def test_api_keys_from_env(self):
        original = os.environ.get("GEMINI_API_KEY")
        try:
            os.environ["GEMINI_API_KEY"] = "test_key"
            config = APIKeysConfig()
            assert config.GEMINI_API_KEY == "test_key"
        finally:
            if original:
                os.environ["GEMINI_API_KEY"] = original
            elif "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]

    def test_all_config_classes(self):
        agent = AgentConfig()
        assert agent.max_steps == 100
        assert agent.verbose is True
        assert agent.debug is False

        docker = DockerConfig()
        assert docker.client_shared_path == "/testbed"

        relentless = RelentlessCodingAgentConfig()
        assert relentless.subtasker_model_name == "claude-opus-4-6"
        assert relentless.max_steps == 200

        kiss = KISSCodingAgentConfig()
        assert kiss.orchestrator_model_name == "claude-sonnet-4-5"
        assert kiss.refiner_model_name == "claude-sonnet-4-5"

        config = Config()
        assert isinstance(config.agent, AgentConfig)
        assert isinstance(config.docker, DockerConfig)


class TestModelHelpers:
    def _create_model(self):
        from kiss.core.models.model import Model

        class ConcreteModel(Model):
            def initialize(self, prompt):
                pass

            def generate(self):
                return "", None

            def generate_and_process_with_tools(self, function_map):
                return [], "", None

            def add_function_results_to_conversation_and_return(self, function_results):
                pass

            def add_message_to_conversation(self, role, content):
                pass

            def extract_input_output_token_counts_from_response(self, response):
                return 0, 0

            def get_embedding(self, text, embedding_model=None):
                return []

        return ConcreteModel("test_model")

    def test_model_basics_and_helpers(self):
        m = self._create_model()
        assert m.model_name == "test_model"
        assert m.model_config == {}

        m.set_usage_info_for_messages("Usage: 100")
        assert m.usage_info_for_messages == "Usage: 100"

        docstring = """Test.\n\nArgs:\n    param1: Description."""
        result = m._parse_docstring_params(docstring)
        assert "param1" in result

    def test_type_to_json_schema_all_types(self):
        m = self._create_model()
        type_map = [
            (str, "string"),
            (int, "integer"),
            (float, "number"),
            (bool, "boolean"),
        ]
        for py_type, expected in type_map:
            result = m._python_type_to_json_schema(py_type)
            assert result["type"] == expected

        result = m._python_type_to_json_schema(list[str])
        assert result["type"] == "array"
        assert result["items"]["type"] == "string"

    def test_function_to_openai_tool(self):
        m = self._create_model()

        def sample(name: str, count: int = 10) -> str:
            """Sample function.\n\nArgs:\n    name: The name.\n    count: Count."""
            return f"{name}: {count}"

        tool = m._function_to_openai_tool(sample)
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "sample"
        assert "name" in tool["function"]["parameters"]["properties"]


class TestModelInfoEdgeCases:
    def test_unknown_model_raises_error(self):
        from kiss.core.kiss_error import KISSError
        from kiss.core.models.model_info import model

        with pytest.raises(KISSError, match="Unknown model name"):
            model("nonexistent-model-xyz")

    def test_get_max_context_length_unknown_model(self):
        from kiss.core.models.model_info import get_max_context_length

        with pytest.raises(KeyError, match="not found in MODEL_INFO"):
            get_max_context_length("nonexistent-model-xyz")

    def test_calculate_cost_unknown_model(self):
        from kiss.core.models.model_info import calculate_cost

        assert calculate_cost("unknown-model-xyz", 1000, 1000) == 0.0

    def test_calculate_cost_known_model(self):
        from kiss.core.models.model_info import calculate_cost

        result = calculate_cost("gpt-4.1-mini", 1000, 1000)
        assert result >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
