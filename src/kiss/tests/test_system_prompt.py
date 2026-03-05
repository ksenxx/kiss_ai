"""Tests for system_prompt parameter in KISSAgent.run()."""

import unittest

from kiss.core.kiss_agent import KISSAgent


class TestSystemPromptInjection(unittest.TestCase):
    """Test that system_prompt is properly injected into model_config."""

    def test_system_prompt_sets_model_config_system_instruction(self) -> None:
        agent = KISSAgent("test")
        # Call run with an invalid model to trigger _reset but capture the model_config
        # We intercept at _reset level by checking the model's config after _reset
        try:
            agent.run(
                model_name="gemini-2.0-flash",
                prompt_template="hello",
                system_prompt="You are a helpful assistant.",
                is_agentic=False,
            )
        except Exception:
            pass
        # After _reset, the model should have system_instruction in its config
        self.assertEqual(
            agent.model.model_config.get("system_instruction"),
            "You are a helpful assistant.",
        )


if __name__ == "__main__":
    unittest.main()
