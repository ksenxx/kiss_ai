"""Test suite for model configuration handling in KISS."""

import unittest
from typing import Any

from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model_info import model as get_model
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel


class CapturingOpenAICompatibleModel(OpenAICompatibleModel):
    """Test subclass that captures API call parameters without making actual calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.captured_kwargs: dict[str, Any] = {}
        self._generate_called = False

    def generate(self) -> tuple[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model_name, "messages": self.conversation}
        if self.model_config:
            kwargs.update(self.model_config)
        self.captured_kwargs = kwargs
        self._generate_called = True
        return "test response", None

    def generate_and_process_with_tools(
        self, function_map: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str, Any]:
        kwargs: dict[str, Any] = {"model": self.model_name, "messages": self.conversation}
        if self.model_config:
            kwargs.update(self.model_config)
        if function_map:
            kwargs["tools"] = list(function_map.keys())
        self.captured_kwargs = kwargs
        self._generate_called = True
        return [], "test response", None


class TestModelConfig(unittest.TestCase):
    def test_model_factory_passes_config(self):
        config = {"temperature": 0.3}
        model_instance = get_model("openrouter/test/model", model_config=config)
        self.assertEqual(model_instance.model_config, config)

    def test_model_factory_returns_anthropic(self):
        config = {"temperature": 0.2}
        model_instance = get_model("claude-opus-4-6", model_config=config)
        self.assertIsInstance(model_instance, AnthropicModel)
        self.assertEqual(model_instance.model_config, config)

    def test_model_config_passed_to_create(self):
        config = {"temperature": 0.5, "top_p": 0.9}
        m = CapturingOpenAICompatibleModel(
            model_name="test-model",
            base_url="http://localhost:1234",
            api_key="sk-test",
            model_config=config,
        )
        m.initialize("Hello")
        m.generate()
        self.assertEqual(m.captured_kwargs.get("temperature"), 0.5)
        self.assertEqual(m.captured_kwargs.get("top_p"), 0.9)
        self.assertEqual(m.captured_kwargs.get("model"), "test-model")
        self.assertTrue(m._generate_called)

    def test_model_config_in_tools_call(self):
        config = {"temperature": 0.7}
        m = CapturingOpenAICompatibleModel(
            model_name="test-model",
            base_url="http://localhost:1234",
            api_key="sk-test",
            model_config=config,
        )
        m.initialize("Hello")
        m.generate_and_process_with_tools({})
        self.assertEqual(m.captured_kwargs.get("temperature"), 0.7)
        self.assertTrue(m._generate_called)

    def test_empty_model_config(self):
        model_no_config = OpenAICompatibleModel(
            model_name="test-model",
            base_url="http://localhost:1234",
            api_key="sk-test",
        )
        self.assertEqual(model_no_config.model_config, {})


if __name__ == "__main__":
    unittest.main()
