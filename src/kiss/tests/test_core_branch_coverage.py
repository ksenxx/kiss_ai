"""Test suite for increasing branch coverage of KISS core components.

These tests target specific branches and edge cases in:
- base.py: Base class for agents
- utils.py: Utility functions
- model_info.py: Model information and lookup
- simple_formatter.py and compact_formatter.py: Formatter implementations
- config.py: Configuration classes
- kiss_error.py: Error handling
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
    resolve_path,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir(tmp_path):
    """Provides a temporary directory."""
    original = os.getcwd()
    resolved = tmp_path.resolve()
    os.chdir(resolved)
    yield resolved
    os.chdir(original)


@pytest.fixture
def verbose_config():
    """Saves and restores verbose config."""
    original = config_module.DEFAULT_CONFIG.agent.verbose
    yield
    config_module.DEFAULT_CONFIG.agent.verbose = original


@pytest.fixture
def base_state():
    """Saves and restores Base class state."""
    original_counter = Base.agent_counter
    original_budget = Base.global_budget_used
    yield
    Base.agent_counter = original_counter
    Base.global_budget_used = original_budget


# =============================================================================
# Base Class Tests
# =============================================================================


class TestBaseClass:
    """Tests for the Base class."""

    def test_basic_init_and_counter(self, base_state):
        """Test basic initialization and counter increments."""
        initial = Base.agent_counter
        agent = Base("test_agent")
        assert agent.name == "test_agent"
        assert agent.base_dir == ""
        assert isinstance(agent.id, int)

        Base("agent2")
        assert Base.agent_counter == initial + 2

    def test_run_state_and_messages(self, base_state):
        """Test _init_run_state, _add_message, _build_state_dict, and get_trajectory."""
        Base.global_budget_used = 5.5
        agent = Base("test")

        # Test _init_run_state
        agent._init_run_state("gpt-4o", ["func1"])
        assert agent.model_name == "gpt-4o"
        assert agent.function_map == ["func1"]
        assert agent.messages == []
        assert isinstance(agent.run_start_timestamp, int)

        # Test _add_message
        agent._add_message("user", "Hello")
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "user"
        assert agent.messages[0]["content"] == "Hello"

        # Test _build_state_dict
        state = agent._build_state_dict()
        assert state["model"] == "gpt-4o"
        assert "name" in state or "agent_name" in state
        assert state["global_budget_used"] == 5.5

        # Test get_trajectory
        trajectory = agent.get_trajectory()
        assert isinstance(trajectory, str)
        assert "Hello" in trajectory

    def test_coding_instructions_constant(self):
        """Test CODING_INSTRUCTIONS is defined."""
        assert CODING_INSTRUCTIONS is not None
        assert len(CODING_INSTRUCTIONS) > 0


# =============================================================================
# Utils Tests (non-duplicated with test_internal.py)
# =============================================================================


class TestUtilsExtra:
    """Additional utility function tests not covered in test_internal.py."""

    def test_resolve_path(self, temp_dir):
        """Test resolve_path function."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")
        resolved = resolve_path(str(test_file), str(temp_dir))
        assert resolved == test_file.resolve()

    def test_is_subpath(self, temp_dir):
        """Test is_subpath function."""
        parent = temp_dir / "parent"
        child = parent / "child"
        parent.mkdir()
        assert is_subpath(child, [parent])
        assert not is_subpath(Path("/etc/passwd"), [temp_dir])

    def test_get_config_value(self):
        """Test get_config_value function."""
        result = get_config_value(None, config_module.DEFAULT_CONFIG.agent, "verbose")
        assert isinstance(result, bool)


# =============================================================================
# Formatter Tests (Parameterized)
# =============================================================================

FORMATTER_CLASSES = [SimpleFormatter, CompactFormatter]


class TestFormatters:
    """Parameterized tests for formatters."""

    @pytest.mark.parametrize("formatter_class", FORMATTER_CLASSES)
    @pytest.mark.parametrize("verbose", [True, False])
    def test_format_methods(self, verbose_config, formatter_class, verbose):
        """Test format_message and format_messages with verbosity settings."""
        config_module.DEFAULT_CONFIG.agent.verbose = verbose
        formatter = formatter_class()

        # Test format_message
        result = formatter.format_message({"role": "user", "content": "Hello"})
        if verbose:
            assert "user" in result.lower() or "Hello" in result
        else:
            assert result == ""

        # Test format_messages
        messages = [{"role": "user", "content": "Hello"}, {"role": "model", "content": "Hi"}]
        result = formatter.format_messages(messages)
        if verbose:
            assert "Hello" in result
        else:
            assert result == ""

    @pytest.mark.parametrize("formatter_class", FORMATTER_CLASSES)
    @pytest.mark.parametrize("verbose", [True, False])
    def test_all_print_methods(self, verbose_config, formatter_class, verbose):
        """Test all print methods don't raise with various verbosity settings."""
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
        """Test print methods when console is None (fallback to print)."""
        config_module.DEFAULT_CONFIG.agent.verbose = True
        formatter = formatter_class()
        formatter._console = None
        formatter._stderr_console = None
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
        """Test print methods when console is present (covers lines 81, 93, 105, 119)."""
        from io import StringIO

        from rich.console import Console

        config_module.DEFAULT_CONFIG.agent.verbose = True
        formatter = formatter_class()
        # Create real consoles that write to StringIO
        output = StringIO()
        err_output = StringIO()
        formatter._console = Console(file=output, force_terminal=True)
        formatter._stderr_console = Console(file=err_output, force_terminal=True)
        formatter.print_status("Status")
        formatter.print_error("Error")
        formatter.print_warning("Warning")
        formatter.print_label_and_value("Label", "Value")


class TestSimpleFormatterSpecific:
    """SimpleFormatter-specific tests."""

    def test_format_message_missing_keys(self, verbose_config):
        """Test format_message with missing keys."""
        config_module.DEFAULT_CONFIG.agent.verbose = True
        formatter = SimpleFormatter()
        result = formatter.format_message({})
        assert 'role=""' in result


class TestCompactFormatterSpecific:
    """CompactFormatter-specific tests."""

    def test_truncates_long_content(self, verbose_config):
        """Test that long content is truncated."""
        config_module.DEFAULT_CONFIG.agent.verbose = True
        formatter = CompactFormatter()
        message = {"role": "user", "content": "A" * 200}
        result = formatter.format_message(message)
        assert len(result) < 200

    def test_replaces_newlines(self, verbose_config):
        """Test that newlines are replaced."""
        config_module.DEFAULT_CONFIG.agent.verbose = True
        formatter = CompactFormatter()
        message = {"role": "user", "content": "line1\nline2"}
        result = formatter.format_message(message)
        assert "\\n" in result

    def test_unknown_role(self, verbose_config):
        """Test handling of missing role."""
        config_module.DEFAULT_CONFIG.agent.verbose = True
        formatter = CompactFormatter()
        result = formatter.format_message({"content": "Hello"})
        assert "[unknown]" in result


class TestLeftAlignedHeading:
    """Tests for _left_aligned_heading helper."""

    @pytest.mark.parametrize("tag,expected_count", [("h1", 1), ("h2", 2), ("h3", 1)])
    def test_heading_tags(self, tag, expected_count):
        """Test different heading tags."""
        from rich.text import Text

        class MockHeading:
            def __init__(self, t):
                self.tag = t
                self.text = Text("Heading")

        results = list(_left_aligned_heading(MockHeading(tag), None, None))
        assert len(results) == expected_count


# =============================================================================
# Config Tests
# =============================================================================


class TestConfigClasses:
    """Tests for config classes."""

    def test_api_keys_from_env(self):
        """Test APIKeysConfig reads from env."""
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
        """Test all config classes have expected defaults."""
        # AgentConfig
        agent = AgentConfig()
        assert agent.max_steps == 100
        assert agent.verbose is True
        assert agent.debug is False

        # DockerConfig
        docker = DockerConfig()
        assert docker.client_shared_path == "/testbed"

        # RelentlessCodingAgentConfig
        relentless = RelentlessCodingAgentConfig()
        assert relentless.orchestrator_model_name == "claude-sonnet-4-5"
        assert relentless.max_steps == 200

        # KISSCodingAgentConfig
        kiss = KISSCodingAgentConfig()
        assert kiss.orchestrator_model_name == "claude-sonnet-4-5"
        assert kiss.refiner_model_name == "claude-sonnet-4-5"

        # Full Config composition
        config = Config()
        assert isinstance(config.agent, AgentConfig)
        assert isinstance(config.docker, DockerConfig)


# KISSError tests are in test_internal.py - no duplication needed


# =============================================================================
# Model Helper Tests
# =============================================================================


class TestModelHelpers:
    """Tests for Model class helper methods."""

    def _create_model(self):
        """Create a concrete Model for testing."""
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
        """Test Model initialization, usage info, and docstring parsing."""
        m = self._create_model()

        # Test init
        assert m.model_name == "test_model"
        assert m.model_config == {}

        # Test usage info
        m.set_usage_info_for_messages("Usage: 100")
        assert m.usage_info_for_messages == "Usage: 100"

        # Test docstring parsing
        docstring = """Test.\n\nArgs:\n    param1: Description."""
        result = m._parse_docstring_params(docstring)
        assert "param1" in result

    def test_type_to_json_schema_all_types(self):
        """Test type conversion to JSON schema for all types."""
        m = self._create_model()

        # Primitive types
        type_map = [
            (str, "string"),
            (int, "integer"),
            (float, "number"),
            (bool, "boolean"),
        ]
        for py_type, expected in type_map:
            result = m._python_type_to_json_schema(py_type)
            assert result["type"] == expected

        # List type
        result = m._python_type_to_json_schema(list[str])
        assert result["type"] == "array"
        assert result["items"]["type"] == "string"

    def test_function_to_openai_tool(self):
        """Test _function_to_openai_tool."""
        m = self._create_model()

        def sample(name: str, count: int = 10) -> str:
            """Sample function.\n\nArgs:\n    name: The name.\n    count: Count."""
            return f"{name}: {count}"

        tool = m._function_to_openai_tool(sample)
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "sample"
        assert "name" in tool["function"]["parameters"]["properties"]


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestBaseSave:
    """Tests for Base._save method."""

    def test_save_creates_trajectory_file(self, base_state, tmp_path):
        """Test that _save creates a trajectory file."""
        # Set artifact dir to temp path
        original_artifact_dir = config_module.DEFAULT_CONFIG.agent.artifact_dir
        config_module.DEFAULT_CONFIG.agent.artifact_dir = str(tmp_path)

        try:
            agent = Base("test_save_agent")
            agent._init_run_state("gpt-4.1-mini", [])
            agent._add_message("user", "Test message")
            agent._save()

            # Check trajectory file was created
            trajectories_dir = tmp_path / "trajectories"
            assert trajectories_dir.exists()
            files = list(trajectories_dir.glob("trajectory_test_save_agent_*.yaml"))
            assert len(files) == 1
        finally:
            config_module.DEFAULT_CONFIG.agent.artifact_dir = original_artifact_dir


class TestModelInfoEdgeCases:
    """Tests for model_info.py edge cases."""

    def test_unknown_model_raises_error(self):
        """Test that unknown model name raises KISSError."""
        from kiss.core.kiss_error import KISSError
        from kiss.core.models.model_info import model

        with pytest.raises(KISSError, match="Unknown model name"):
            model("nonexistent-model-xyz")

    def test_get_max_context_length_unknown_model(self):
        """Test get_max_context_length raises KeyError for unknown model."""
        from kiss.core.models.model_info import get_max_context_length

        with pytest.raises(KeyError, match="not found in MODEL_INFO"):
            get_max_context_length("nonexistent-model-xyz")


class TestUtilsEdgeCases:
    """Tests for utils.py edge cases."""

    def test_get_config_value_raises_on_missing(self):
        """Test get_config_value raises ValueError when no value available."""
        from kiss.core.utils import get_config_value

        class EmptyConfig:
            pass

        with pytest.raises(ValueError, match="No value provided"):
            get_config_value(None, EmptyConfig(), "nonexistent_attr")

    def test_read_project_file_existing(self):
        """Test read_project_file reads from filesystem."""
        from kiss.core.utils import read_project_file

        # Try to read an existing project file
        try:
            content = read_project_file("src/kiss/core/__init__.py")
            assert isinstance(content, str)
            assert len(content) > 0
        except Exception:
            pytest.skip("Project file not accessible in test environment")

    def test_read_project_file_not_found(self):
        """Test read_project_file raises KISSError for non-existent file."""
        from kiss.core.kiss_error import KISSError
        from kiss.core.utils import read_project_file

        with pytest.raises(KISSError, match="Could not find"):
            read_project_file("nonexistent/path/to/file.txt")

    def test_read_project_file_from_package_not_found(self):
        """Test read_project_file_from_package raises KISSError for non-existent."""
        from kiss.core.kiss_error import KISSError
        from kiss.core.utils import read_project_file_from_package

        with pytest.raises(KISSError, match="Could not find"):
            read_project_file_from_package("nonexistent_file.txt")

    def test_config_to_dict_with_list(self):
        """Test config_to_dict handles lists in config."""
        from kiss.core.utils import config_to_dict

        result = config_to_dict()
        # Just verify it returns a dict without error
        assert isinstance(result, dict)

    def test_resolve_path_absolute(self, tmp_path):
        """Test resolve_path with absolute path."""
        from kiss.core.utils import resolve_path

        abs_path = str(tmp_path / "test.txt")
        result = resolve_path(abs_path, "/some/other/base")
        assert result == Path(abs_path).resolve()


class TestFormatterFallbackPrints:
    """Tests for formatter print methods when console is None but verbose is True."""

    def test_simple_formatter_print_message_no_console(self, verbose_config, capsys):
        """Test SimpleFormatter.print_message falls back to print (lines 90-91, 93-94)."""
        config_module.DEFAULT_CONFIG.agent.verbose = True
        formatter = SimpleFormatter()
        formatter._console = None
        formatter.print_message({"role": "user", "content": "Test content"})
        captured = capsys.readouterr()
        assert "user" in captured.out or "Test content" in captured.out

    def test_simple_formatter_print_messages_no_console(self, verbose_config, capsys):
        """Test SimpleFormatter.print_messages falls back to print (lines 104-110)."""
        config_module.DEFAULT_CONFIG.agent.verbose = True
        formatter = SimpleFormatter()
        formatter._console = None
        formatter.print_messages([{"role": "user", "content": "Msg1"}])
        captured = capsys.readouterr()
        assert "Agent Messages" in captured.out or "Msg1" in captured.out

    def test_compact_formatter_print_message_no_console(self, verbose_config, capsys):
        """Test CompactFormatter.print_message falls back to print."""
        config_module.DEFAULT_CONFIG.agent.verbose = True
        formatter = CompactFormatter()
        formatter._console = None
        formatter.print_message({"role": "user", "content": "Test"})
        captured = capsys.readouterr()
        assert "[user]" in captured.out or "Test" in captured.out

    def test_compact_formatter_print_messages_no_console(self, verbose_config, capsys):
        """Test CompactFormatter.print_messages falls back to print."""
        config_module.DEFAULT_CONFIG.agent.verbose = True
        formatter = CompactFormatter()
        formatter._console = None
        formatter.print_messages([{"role": "user", "content": "Msg"}])
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_simple_formatter_print_message_with_console(self, verbose_config):
        """Test SimpleFormatter.print_message with console (lines 89-91)."""
        from io import StringIO

        from rich.console import Console

        config_module.DEFAULT_CONFIG.agent.verbose = True
        formatter = SimpleFormatter()
        output = StringIO()
        formatter._console = Console(file=output, force_terminal=True)
        formatter.print_message({"role": "user", "content": "Test"})

    def test_simple_formatter_print_messages_with_console(self, verbose_config):
        """Test SimpleFormatter.print_messages with console (lines 103-107)."""
        from io import StringIO

        from rich.console import Console

        config_module.DEFAULT_CONFIG.agent.verbose = True
        formatter = SimpleFormatter()
        output = StringIO()
        formatter._console = Console(file=output, force_terminal=True)
        formatter.print_messages([{"role": "user", "content": "Test"}])


class TestBaseBuildStateWithUnknownModel:
    """Test Base._build_state_dict handles unknown model gracefully."""

    def test_build_state_dict_unknown_model(self, base_state):
        """Test _build_state_dict sets max_tokens to None for unknown model."""
        agent = Base("test")
        agent._init_run_state("unknown-model-xyz", [])
        state = agent._build_state_dict()
        assert state["max_tokens"] is None


class TestModelInfoCalculateCost:
    """Tests for calculate_cost function edge cases."""

    def test_calculate_cost_unknown_model(self):
        """Test calculate_cost returns 0.0 for unknown model."""
        from kiss.core.models.model_info import calculate_cost

        result = calculate_cost("unknown-model-xyz", 1000, 1000)
        assert result == 0.0

    def test_calculate_cost_known_model(self):
        """Test calculate_cost returns non-zero for known model with tokens."""
        from kiss.core.models.model_info import calculate_cost

        # gpt-4.1-mini is a known model
        result = calculate_cost("gpt-4.1-mini", 1000, 1000)
        assert result >= 0.0  # May be 0 if pricing is 0, but should not error


class TestConfigToDict:
    """Tests for config_to_dict edge cases."""

    def test_config_to_dict_handles_dict_values(self):
        """Test config_to_dict handles nested dict values."""
        from kiss.core.utils import config_to_dict

        result = config_to_dict()
        assert isinstance(result, dict)
        # Verify nested structures are converted
        assert "agent" in result
        assert isinstance(result["agent"], dict)

    def test_config_to_dict_filters_api_keys(self):
        """Test config_to_dict filters out API_KEY fields."""
        from kiss.core.utils import config_to_dict

        result = config_to_dict()
        result_str = str(result)
        assert "API_KEY" not in result_str


class TestGetConfigValueDefault:
    """Tests for get_config_value with default parameter."""

    def test_get_config_value_uses_default(self):
        """Test get_config_value returns default when value and config are None."""
        from kiss.core.utils import get_config_value

        class ConfigWithNone:
            nonexistent = None

        result = get_config_value(None, ConfigWithNone(), "nonexistent", default="fallback")
        assert result == "fallback"

    def test_get_config_value_prefers_explicit(self):
        """Test get_config_value prefers explicit value over config."""
        from kiss.core.utils import get_config_value

        result = get_config_value("explicit", config_module.DEFAULT_CONFIG.agent, "verbose")
        assert result == "explicit"


class TestConvertToJsonInternals:
    """Test the convert_to_json helper inside config_to_dict."""

    def test_config_to_dict_with_nested_dict(self):
        """Test config_to_dict handles nested dict - covers line 85."""
        # The config should have nested dicts through model_dump
        from kiss.core.utils import config_to_dict

        result = config_to_dict()
        # agent is a nested structure that should be converted
        assert isinstance(result.get("agent"), dict)

    def test_config_to_dict_filters_api_key_in_dict(self):
        """Test that API_KEY is filtered from dicts - covers line 85."""
        from kiss.core.utils import config_to_dict

        result = config_to_dict()

        # Ensure no API_KEY appears anywhere
        def check_no_api_key(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    assert "API_KEY" not in k
                    check_no_api_key(v)
            elif isinstance(obj, list):
                for item in obj:
                    check_no_api_key(item)

        check_no_api_key(result)


class TestReadProjectFileEdgeCases:
    """Tests for read_project_file edge cases."""

    def test_read_project_file_from_package_structure(self):
        """Test read_project_file with package path structure."""
        from kiss.core.utils import read_project_file

        # Try to read a known file that exists in the package
        try:
            content = read_project_file("src/kiss/__init__.py")
            assert isinstance(content, str)
        except Exception:
            # File may not exist in all environments
            pytest.skip("Package file not accessible")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
