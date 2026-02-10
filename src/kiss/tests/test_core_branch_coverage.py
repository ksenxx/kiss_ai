"""Test suite for increasing branch coverage of KISS core components.

These tests target specific branches and edge cases in:
- base.py: Base class for agents
- utils.py: Utility functions
- model_info.py: Model information and lookup
- simple_formatter.py and compact_formatter.py: Formatter implementations
"""

import pytest

from kiss.core import config as config_module
from kiss.core.base import Base
from kiss.core.compact_formatter import CompactFormatter
from kiss.core.simple_formatter import SimpleFormatter, _left_aligned_heading
from kiss.core.utils import (
    get_config_value,
    read_project_file,
    read_project_file_from_package,
)

FORMATTER_CLASSES = [SimpleFormatter, CompactFormatter]


class TestBaseClass:
    @pytest.fixture(autouse=True)
    def base_state(self):
        original_counter = Base.agent_counter
        original_budget = Base.global_budget_used
        yield
        Base.agent_counter = original_counter
        Base.global_budget_used = original_budget

    def test_build_state_dict_unknown_model(self):
        agent = Base("test")
        agent._init_run_state("unknown-model-xyz", [])
        state = agent._build_state_dict()
        assert state["max_tokens"] is None


class TestUtils:
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

    def test_read_project_file_not_found(self):
        from kiss.core.kiss_error import KISSError

        with pytest.raises(KISSError, match="Could not find"):
            read_project_file("nonexistent/path/to/file.txt")

    def test_read_project_file_from_package_not_found(self):
        from kiss.core.kiss_error import KISSError

        with pytest.raises(KISSError, match="Could not find"):
            read_project_file_from_package("nonexistent_file.txt")


class TestFormatters:
    def test_format_methods_false_simple_formatter(self, verbose_config):
        config_module.DEFAULT_CONFIG.agent.verbose = False
        formatter = SimpleFormatter()

        result = formatter.format_message({"role": "user", "content": "Hello"})
        assert result == ""

        messages = [{"role": "user", "content": "Hello"}, {"role": "model", "content": "Hi"}]
        result = formatter.format_messages(messages)
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


class TestModelInfoEdgeCases:
    def test_unknown_model_raises_error(self):
        from kiss.core.kiss_error import KISSError
        from kiss.core.models.model_info import model

        with pytest.raises(KISSError, match="Unknown model name"):
            model("nonexistent-model-xyz")

    def test_calculate_cost_unknown_model(self):
        from kiss.core.models.model_info import calculate_cost

        assert calculate_cost("unknown-model-xyz", 1000, 1000) == 0.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
