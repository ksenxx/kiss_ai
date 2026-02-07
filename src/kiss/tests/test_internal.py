"""Test suite for internal KISS components that don't require API calls."""

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
    def test_get_template_field_names(self):
        fields = get_template_field_names("Hello {name}, your score is {score}.")
        self.assertIn("name", fields)
        self.assertIn("score", fields)
        self.assertEqual(len(fields), 2)
        self.assertEqual(get_template_field_names("Hello, world!"), [])

    def test_add_prefix_to_each_line(self):
        self.assertEqual(
            add_prefix_to_each_line("line1\nline2\nline3", "> "),
            "> line1\n> line2\n> line3",
        )
        self.assertEqual(add_prefix_to_each_line("single line", ">> "), ">> single line")

    def test_config_to_dict(self):
        config_dict = config_to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn("agent", config_dict)
        self.assertIsInstance(config_dict["agent"], dict)

        result_str = str(config_dict)
        self.assertNotIn("GEMINI_API_KEY", result_str)
        self.assertNotIn("OPENAI_API_KEY", result_str)

        self.assertIsInstance(config_dict.get("agent", {}).get("max_steps"), int)
        self.assertIsInstance(config_dict.get("agent", {}).get("verbose"), bool)

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
        result = finish("success", "Task completed", "42")
        self.assertIn("status", result)
        self.assertIn("success", result)
        self.assertIn("analysis", result)
        self.assertIn("Task completed", result)
        self.assertIn("result", result)
        self.assertIn("42", result)

    def test_fc_reads_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for fc function")
            temp_path = f.name
        try:
            self.assertEqual(fc(temp_path), "Test content for fc function")
        finally:
            os.unlink(temp_path)


class TestKISSError(unittest.TestCase):
    def test_kiss_error_message(self):
        error = KISSError("Test error message")
        error_str = str(error)
        self.assertIn("KISS Error", error_str)
        self.assertIn("Test error message", error_str)

    def test_kiss_error_inheritance(self):
        self.assertIsInstance(KISSError("Test"), ValueError)

    def test_kiss_error_debug_mode(self):
        from kiss.core.config import DEFAULT_CONFIG

        original_debug = DEFAULT_CONFIG.agent.debug
        DEFAULT_CONFIG.agent.debug = False
        try:
            error = KISSError("Test message")
            self.assertIn("Test message", str(error))
        finally:
            DEFAULT_CONFIG.agent.debug = original_debug


class TestConfigBuilder(unittest.TestCase):
    def test_add_config_basic(self):
        from pydantic import BaseModel, Field

        from kiss.core import config as config_module
        from kiss.core.config_builder import add_config

        class TestConfig(BaseModel):
            test_value: str = Field(default="test")
            test_int: int = Field(default=42)

        original_config = config_module.DEFAULT_CONFIG
        try:
            add_config("test_config", TestConfig)
            self.assertIsNotNone(config_module.DEFAULT_CONFIG)
        finally:
            config_module.DEFAULT_CONFIG = original_config

    def test_add_config_with_nested_model(self):
        from pydantic import BaseModel, Field

        from kiss.core import config as config_module
        from kiss.core.config_builder import add_config

        class InnerConfig(BaseModel):
            inner_value: str = Field(default="inner")

        class OuterConfig(BaseModel):
            outer_value: str = Field(default="outer")
            inner: InnerConfig = Field(default_factory=InnerConfig)

        original_config = config_module.DEFAULT_CONFIG
        try:
            add_config("outer_config", OuterConfig)
            self.assertIsNotNone(config_module.DEFAULT_CONFIG)
        finally:
            config_module.DEFAULT_CONFIG = original_config

    def test_add_model_arguments_with_types(self):
        from argparse import ArgumentParser
        from typing import Any

        from pydantic import BaseModel, Field

        from kiss.core.config_builder import _add_model_arguments

        class ConfigWithTypes(BaseModel):
            optional_str: str | None = Field(default=None)
            optional_int: int | None = Field(default=None)
            union_type: str | int = Field(default="test")
            any_field: Any = Field(default=None)

        parser = ArgumentParser()
        _add_model_arguments(parser, ConfigWithTypes)
        args, _ = parser.parse_known_args([])
        self.assertIsNotNone(args)

    def test_flat_to_nested_dict(self):
        from pydantic import BaseModel, Field

        from kiss.core.config_builder import _flat_to_nested_dict

        class InnerModel(BaseModel):
            value: str = Field(default="test")

        class OuterModel(BaseModel):
            inner: InnerModel = Field(default_factory=InnerModel)

        class SimpleModel(BaseModel):
            value: str = Field(default="test")
            number: int = Field(default=0)

        result_prefix = _flat_to_nested_dict({"inner__value": "custom_value"}, OuterModel)
        self.assertIn("inner", result_prefix)
        self.assertEqual(result_prefix["inner"]["value"], "custom_value")

        self.assertEqual(_flat_to_nested_dict({}, SimpleModel), {})

        result_values = _flat_to_nested_dict({"value": "custom", "number": 42}, SimpleModel)
        self.assertEqual(result_values["value"], "custom")
        self.assertEqual(result_values["number"], 42)


if __name__ == "__main__":
    unittest.main()
