# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Test suite for internal KISS components that don't require API calls.

These tests verify various internal components like formatters, utilities,
config builders, and error classes without making real API calls.
"""

import os
import tempfile
import unittest

from kiss.core.kiss_error import KISSError
from kiss.core.utils import (
    add_prefix_to_each_line,
    config_to_dict,
    fc,
    finish,
    get_template_field_names,
)


class TestUtilsFunctions(unittest.TestCase):
    """Tests for utility functions."""

    def test_get_template_field_names(self):
        """Test get_template_field_names extracts field names."""
        # With fields
        text = "Hello {name}, your score is {score}."
        fields = get_template_field_names(text)
        self.assertIn("name", fields)
        self.assertIn("score", fields)
        self.assertEqual(len(fields), 2)

        # Without fields
        text_no_fields = "Hello, world!"
        fields_empty = get_template_field_names(text_no_fields)
        self.assertEqual(fields_empty, [])

    def test_add_prefix_to_each_line(self):
        """Test add_prefix_to_each_line adds prefix correctly."""
        # Multiple lines
        text = "line1\nline2\nline3"
        result = add_prefix_to_each_line(text, "> ")
        self.assertEqual(result, "> line1\n> line2\n> line3")

        # Single line
        single = "single line"
        result_single = add_prefix_to_each_line(single, ">> ")
        self.assertEqual(result_single, ">> single line")

    def test_config_to_dict(self):
        """Test config_to_dict returns a properly structured dictionary."""
        config_dict = config_to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn("agent", config_dict)

        # API keys should be filtered out
        result_str = str(config_dict)
        self.assertNotIn("GEMINI_API_KEY", result_str)
        self.assertNotIn("OPENAI_API_KEY", result_str)

        # Verify primitive types are preserved
        self.assertIsInstance(config_dict.get("agent", {}).get("max_steps"), int)
        self.assertIsInstance(config_dict.get("agent", {}).get("verbose"), bool)

        # Verify no API_KEY fields in nested structure
        def check_no_api_keys(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    self.assertNotIn("API_KEY", k)
                    check_no_api_keys(v)
            elif isinstance(d, list):
                for item in d:
                    check_no_api_keys(item)

        check_no_api_keys(config_dict)

    def test_finish_function(self):
        """Test finish utility function."""
        result = finish("success", "Task completed", "42")
        self.assertIn("status", result)
        self.assertIn("success", result)
        self.assertIn("analysis", result)
        self.assertIn("Task completed", result)
        self.assertIn("result", result)
        self.assertIn("42", result)

    def test_fc_reads_file(self):
        """Test fc function reads file content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for fc function")
            temp_path = f.name

        try:
            content = fc(temp_path)
            self.assertEqual(content, "Test content for fc function")
        finally:
            os.unlink(temp_path)


class TestKISSError(unittest.TestCase):
    """Tests for KISSError class."""

    def test_kiss_error_message(self):
        """Test KISSError stores and returns message."""
        error = KISSError("Test error message")
        error_str = str(error)
        self.assertIn("KISS Error", error_str)
        self.assertIn("Test error message", error_str)

    def test_kiss_error_inheritance(self):
        """Test KISSError is a ValueError."""
        error = KISSError("Test")
        self.assertIsInstance(error, ValueError)

    def test_kiss_error_debug_mode(self):
        """Test KISSError message in different debug modes."""
        from kiss.core.config import DEFAULT_CONFIG

        original_debug = DEFAULT_CONFIG.agent.debug
        DEFAULT_CONFIG.agent.debug = False
        try:
            error = KISSError("Test message")
            self.assertIn("Test message", str(error))
        finally:
            DEFAULT_CONFIG.agent.debug = original_debug


class TestConfigBuilder(unittest.TestCase):
    """Tests for config_builder.py functions."""

    def test_add_config_basic(self):
        """Test add_config creates a configuration."""
        from pydantic import BaseModel, Field

        from kiss.core import config as config_module
        from kiss.core.config_builder import add_config

        class TestConfig(BaseModel):
            test_value: str = Field(default="test", description="A test value")
            test_int: int = Field(default=42, description="A test integer")

        original_config = config_module.DEFAULT_CONFIG

        try:
            add_config("test_config", TestConfig)
            self.assertIsNotNone(config_module.DEFAULT_CONFIG)
        finally:
            config_module.DEFAULT_CONFIG = original_config

    def test_add_config_with_nested_model(self):
        """Test add_config with nested BaseModel."""
        from pydantic import BaseModel, Field

        from kiss.core import config as config_module
        from kiss.core.config_builder import add_config

        class InnerConfig(BaseModel):
            inner_value: str = Field(default="inner", description="Inner value")

        class OuterConfig(BaseModel):
            outer_value: str = Field(default="outer", description="Outer value")
            inner: InnerConfig = Field(default_factory=InnerConfig)

        original_config = config_module.DEFAULT_CONFIG

        try:
            add_config("outer_config", OuterConfig)
            self.assertIsNotNone(config_module.DEFAULT_CONFIG)
        finally:
            config_module.DEFAULT_CONFIG = original_config

    def test_add_model_arguments_with_types(self):
        """Test _add_model_arguments handles various types."""
        from argparse import ArgumentParser
        from typing import Any

        from pydantic import BaseModel, Field

        from kiss.core.config_builder import _add_model_arguments

        class ConfigWithTypes(BaseModel):
            optional_str: str | None = Field(default=None, description="Optional string")
            optional_int: int | None = Field(default=None, description="Optional int")
            union_type: str | int = Field(default="test", description="Union type")
            any_field: Any = Field(default=None, description="Any type field")

        parser = ArgumentParser()
        _add_model_arguments(parser, ConfigWithTypes)
        args, _ = parser.parse_known_args([])
        self.assertIsNotNone(args)

    def test_flat_to_nested_dict(self):
        """Test _flat_to_nested_dict with various inputs."""
        from pydantic import BaseModel, Field

        from kiss.core.config_builder import _flat_to_nested_dict

        class InnerModel(BaseModel):
            value: str = Field(default="test")

        class OuterModel(BaseModel):
            inner: InnerModel = Field(default_factory=InnerModel)

        class SimpleModel(BaseModel):
            value: str = Field(default="test")
            number: int = Field(default=0)

        # With prefix
        flat_prefix = {"inner__value": "custom_value"}
        result_prefix = _flat_to_nested_dict(flat_prefix, OuterModel)
        self.assertIn("inner", result_prefix)
        self.assertEqual(result_prefix["inner"]["value"], "custom_value")

        # Empty
        flat_empty = {}
        result_empty = _flat_to_nested_dict(flat_empty, SimpleModel)
        self.assertEqual(result_empty, {})

        # With values
        flat_values = {"value": "custom", "number": 42}
        result_values = _flat_to_nested_dict(flat_values, SimpleModel)
        self.assertEqual(result_values["value"], "custom")
        self.assertEqual(result_values["number"], 42)


if __name__ == "__main__":
    unittest.main()
