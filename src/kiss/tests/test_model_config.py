"""Test suite for model configuration handling in KISS."""

import unittest

from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model_info import model as get_model


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


if __name__ == "__main__":
    unittest.main()
