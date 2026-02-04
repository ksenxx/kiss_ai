"""Test suite for model configuration handling in KISS.

These tests verify that model configurations are properly passed through
the model factory and correctly applied during API calls.
"""

import unittest
from unittest.mock import MagicMock

from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model_info import model as get_model
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel


class TestModelConfig(unittest.TestCase):
    """Tests for model configuration passing and application."""

    def test_model_factory_passes_config(self):
        """Test that the model factory correctly passes config to OpenAICompatibleModel.

        Verifies that when creating a model through the factory with model_config,
        the configuration is stored in the model instance.

        Returns:
            None. Uses assertions to verify config is properly stored.
        """
        config = {"temperature": 0.3}
        # Use a model name that triggers OpenAICompatibleModel, e.g. openrouter/test
        model_instance = get_model("openrouter/test/model", model_config=config)

        self.assertEqual(model_instance.model_config, config)

    def test_model_factory_returns_anthropic(self):
        """Test that the model factory returns AnthropicModel for Claude models.

        Verifies that Claude model names are routed to the AnthropicModel class
        and that model_config is properly stored.

        Returns:
            None. Uses assertions to verify model type and config storage.
        """
        config = {"temperature": 0.2}
        model_instance = get_model("claude-opus-4-5", model_config=config)
        self.assertIsInstance(model_instance, AnthropicModel)
        self.assertEqual(model_instance.model_config, config)

    def test_model_config_passed_to_create(self):
        """Test that model_config parameters are passed to the API create call.

        Verifies that configuration values like temperature and top_p are
        included in the kwargs when calling client.chat.completions.create.

        Returns:
            None. Uses mocked client to verify API call parameters.
        """
        config = {"temperature": 0.5, "top_p": 0.9}
        model = OpenAICompatibleModel(
            model_name="test-model",
            base_url="http://localhost:1234",
            api_key="sk-test",
            model_config=config,
        )

        # Mock the client and initialize
        model.initialize("Hello")
        model.client = MagicMock()
        model.client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="response", tool_calls=None))
        ]

        # Call generate
        model.generate()

        # Check if create was called with config
        call_kwargs = model.client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs.get("temperature"), 0.5)
        self.assertEqual(call_kwargs.get("top_p"), 0.9)
        self.assertEqual(call_kwargs.get("model"), "test-model")

    def test_model_config_in_tools_call(self):
        """Test that model_config is applied during tool-enabled API calls.

        Verifies that configuration parameters are passed through when using
        generate_and_process_with_tools method.

        Returns:
            None. Uses mocked client to verify config parameters in tool calls.
        """
        config = {"temperature": 0.7}
        model = OpenAICompatibleModel(
            model_name="test-model",
            base_url="http://localhost:1234",
            api_key="sk-test",
            model_config=config,
        )

        model.initialize("Hello")
        model.client = MagicMock()
        model.client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="response", tool_calls=None))
        ]

        # Call generate_and_process_with_tools with empty function map
        model.generate_and_process_with_tools({})

        call_kwargs = model.client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs.get("temperature"), 0.7)


if __name__ == "__main__":
    unittest.main()
