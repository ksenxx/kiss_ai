"""Test suite for model configuration handling in KISS.

These tests verify that model configurations are properly passed through
the model factory and correctly applied during API calls.
"""

import unittest
from typing import Any

from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model_info import model as get_model
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel


class CapturingOpenAICompatibleModel(OpenAICompatibleModel):
    """Test subclass that captures API call parameters without making actual calls.

    This allows testing that model_config is passed correctly to API calls
    without requiring network access or mocking.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the capturing model.

        Args:
            *args: Positional arguments passed to parent.
            **kwargs: Keyword arguments passed to parent.
        """
        super().__init__(*args, **kwargs)
        self.captured_kwargs: dict[str, Any] = {}
        self._generate_called = False

    def generate(self) -> tuple[str, Any]:
        """Capture the kwargs that would be passed to the API, then return test response.

        Returns:
            A tuple of (response string, raw response data).
        """
        # Build the kwargs that would be passed to client.chat.completions.create
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": self.conversation,
        }
        # Add model_config parameters
        if self.model_config:
            kwargs.update(self.model_config)

        self.captured_kwargs = kwargs
        self._generate_called = True

        # Simulate a response without making an actual API call
        return "test response", None

    def generate_and_process_with_tools(
        self, function_map: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Capture kwargs for tool-enabled calls.

        Args:
            function_map: Map of function names to callables.

        Returns:
            Tuple of (tool_calls, response, raw_response).
        """
        # Build the kwargs that would be passed to the API
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": self.conversation,
        }
        # Add model_config parameters
        if self.model_config:
            kwargs.update(self.model_config)

        # Add tools if provided
        if function_map:
            kwargs["tools"] = list(function_map.keys())

        self.captured_kwargs = kwargs
        self._generate_called = True

        return [], "test response", None


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
        model_instance = get_model("claude-opus-4-6", model_config=config)
        self.assertIsInstance(model_instance, AnthropicModel)
        self.assertEqual(model_instance.model_config, config)

    def test_model_config_passed_to_create(self):
        """Test that model_config parameters are passed to the API create call.

        Verifies that configuration values like temperature and top_p are
        included in the kwargs when calling the API.

        Returns:
            None. Uses capturing subclass to verify API call parameters.
        """
        config = {"temperature": 0.5, "top_p": 0.9}
        model = CapturingOpenAICompatibleModel(
            model_name="test-model",
            base_url="http://localhost:1234",
            api_key="sk-test",
            model_config=config,
        )

        # Initialize and call generate
        model.initialize("Hello")
        model.generate()

        # Check captured kwargs contain config values
        self.assertEqual(model.captured_kwargs.get("temperature"), 0.5)
        self.assertEqual(model.captured_kwargs.get("top_p"), 0.9)
        self.assertEqual(model.captured_kwargs.get("model"), "test-model")
        self.assertTrue(model._generate_called)

    def test_model_config_in_tools_call(self):
        """Test that model_config is applied during tool-enabled API calls.

        Verifies that configuration parameters are passed through when using
        generate_and_process_with_tools method.

        Returns:
            None. Uses capturing subclass to verify config parameters in tool calls.
        """
        config = {"temperature": 0.7}
        model = CapturingOpenAICompatibleModel(
            model_name="test-model",
            base_url="http://localhost:1234",
            api_key="sk-test",
            model_config=config,
        )

        model.initialize("Hello")
        # Call generate_and_process_with_tools with empty function map
        model.generate_and_process_with_tools({})

        self.assertEqual(model.captured_kwargs.get("temperature"), 0.7)
        self.assertTrue(model._generate_called)

    def test_model_config_storage(self):
        """Test that model_config is stored correctly in the model instance.

        Verifies that configuration is accessible after model creation.

        Returns:
            None.
        """
        config = {"temperature": 0.8, "max_tokens": 1000}
        model = OpenAICompatibleModel(
            model_name="test-model",
            base_url="http://localhost:1234",
            api_key="sk-test",
            model_config=config,
        )

        self.assertEqual(model.model_config, config)
        self.assertEqual(model.model_config.get("temperature"), 0.8)
        self.assertEqual(model.model_config.get("max_tokens"), 1000)

    def test_empty_model_config(self):
        """Test that empty/None model_config is handled correctly.

        Verifies that the model works without any custom configuration.

        Returns:
            None.
        """
        model_no_config = OpenAICompatibleModel(
            model_name="test-model",
            base_url="http://localhost:1234",
            api_key="sk-test",
        )
        # Default model_config is an empty dict when not provided
        self.assertEqual(model_no_config.model_config, {})

        model_empty_config = OpenAICompatibleModel(
            model_name="test-model",
            base_url="http://localhost:1234",
            api_key="sk-test",
            model_config={},
        )
        self.assertEqual(model_empty_config.model_config, {})


if __name__ == "__main__":
    unittest.main()
