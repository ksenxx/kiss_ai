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

    def test_empty_system_prompt_does_not_override_model_config(self) -> None:
        agent = KISSAgent("test")
        try:
            agent.run(
                model_name="gemini-2.0-flash",
                prompt_template="hello",
                system_prompt="",
                model_config={"system_instruction": "from config"},
                is_agentic=False,
            )
        except Exception:
            pass
        self.assertEqual(
            agent.model.model_config.get("system_instruction"),
            "from config",
        )

    def test_system_prompt_overrides_model_config_system_instruction(self) -> None:
        agent = KISSAgent("test")
        try:
            agent.run(
                model_name="gemini-2.0-flash",
                prompt_template="hello",
                system_prompt="override prompt",
                model_config={"system_instruction": "original"},
                is_agentic=False,
            )
        except Exception:
            pass
        self.assertEqual(
            agent.model.model_config.get("system_instruction"),
            "override prompt",
        )

    def test_no_system_prompt_no_model_config(self) -> None:
        agent = KISSAgent("test")
        try:
            agent.run(
                model_name="gemini-2.0-flash",
                prompt_template="hello",
                is_agentic=False,
            )
        except Exception:
            pass
        self.assertNotIn("system_instruction", agent.model.model_config)

    def test_system_prompt_does_not_mutate_original_model_config(self) -> None:
        original_config = {"temperature": 0.5}
        agent = KISSAgent("test")
        try:
            agent.run(
                model_name="gemini-2.0-flash",
                prompt_template="hello",
                system_prompt="injected",
                model_config=original_config,
                is_agentic=False,
            )
        except Exception:
            pass
        # Original dict should not be modified
        self.assertNotIn("system_instruction", original_config)
        # But model should have it
        self.assertEqual(
            agent.model.model_config.get("system_instruction"),
            "injected",
        )

    def test_system_prompt_preserves_other_model_config_keys(self) -> None:
        agent = KISSAgent("test")
        try:
            agent.run(
                model_name="gemini-2.0-flash",
                prompt_template="hello",
                system_prompt="my prompt",
                model_config={"temperature": 0.7, "max_tokens": 100},
                is_agentic=False,
            )
        except Exception:
            pass
        self.assertEqual(agent.model.model_config["system_instruction"], "my prompt")
        self.assertEqual(agent.model.model_config["temperature"], 0.7)
        self.assertEqual(agent.model.model_config["max_tokens"], 100)


if __name__ == "__main__":
    unittest.main()
