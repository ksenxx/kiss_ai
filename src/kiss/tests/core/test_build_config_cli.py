"""Integration tests for build_config() CLI argument parsing.

Verifies that base Config fields in config.py are accessible via
command-line arguments through the build_config() function.
"""

import sys
import unittest

from pydantic import BaseModel, Field

from kiss.core import config as config_module
from kiss.core.config import Config
from kiss.core.config_builder import add_config, build_config


class TestBuildConfigCLI(unittest.TestCase):
    """Integration tests for build_config() overriding base Config fields via CLI."""

    def setUp(self) -> None:
        self.original_config = config_module.DEFAULT_CONFIG
        self.original_argv = sys.argv

    def tearDown(self) -> None:
        sys.argv = self.original_argv
        config_module.DEFAULT_CONFIG = self.original_config

    def test_max_budget_override_with_dashes(self) -> None:
        """--max-budget should override the default max_budget."""
        sys.argv = ["test", "--max-budget", "500.0"]
        build_config()
        self.assertEqual(config_module.DEFAULT_CONFIG.max_budget, 500.0)

    def test_max_budget_override_with_underscores(self) -> None:
        """--max_budget (underscore variant) should also work."""
        sys.argv = ["test", "--max_budget", "42.5"]
        build_config()
        self.assertEqual(config_module.DEFAULT_CONFIG.max_budget, 42.5)

    def test_no_args_preserves_defaults(self) -> None:
        """Calling build_config with no CLI args preserves default values."""
        sys.argv = ["test"]
        build_config()
        self.assertEqual(config_module.DEFAULT_CONFIG.max_budget, Config().max_budget)

    def test_api_key_override(self) -> None:
        """--GEMINI-API-KEY should override the Gemini API key."""
        sys.argv = ["test", "--GEMINI-API-KEY", "test-key-123"]
        build_config()
        self.assertEqual(config_module.DEFAULT_CONFIG.GEMINI_API_KEY, "test-key-123")

    def test_multiple_overrides(self) -> None:
        """Multiple CLI flags should all take effect."""
        sys.argv = [
            "test",
            "--max-budget", "99.9",
            "--OPENAI-API-KEY", "sk-test",
        ]
        build_config()
        self.assertEqual(config_module.DEFAULT_CONFIG.max_budget, 99.9)
        self.assertEqual(config_module.DEFAULT_CONFIG.OPENAI_API_KEY, "sk-test")

    def test_unknown_args_ignored(self) -> None:
        """Unknown CLI args should be silently ignored (parse_known_args)."""
        sys.argv = ["test", "--max-budget", "77.0", "--unknown-flag", "val"]
        build_config()
        self.assertEqual(config_module.DEFAULT_CONFIG.max_budget, 77.0)

    def test_build_config_preserves_extended_fields(self) -> None:
        """build_config after add_config should preserve extension fields."""

        class ExtraConfig(BaseModel):
            extra_val: int = Field(default=10)

        sys.argv = ["test"]
        add_config("extra", ExtraConfig)
        self.assertEqual(config_module.DEFAULT_CONFIG.extra.extra_val, 10)  # type: ignore[union-attr]

        # Now override a base field via build_config
        sys.argv = ["test", "--max-budget", "333.0"]
        build_config()
        self.assertEqual(config_module.DEFAULT_CONFIG.max_budget, 333.0)
        # Extension field should still be accessible
        self.assertEqual(config_module.DEFAULT_CONFIG.extra.extra_val, 10)  # type: ignore[union-attr]

    def test_build_config_overrides_extended_sub_field(self) -> None:
        """build_config should also parse CLI args for extension sub-fields."""

        class SubConfig(BaseModel):
            depth: int = Field(default=3)

        sys.argv = ["test"]
        add_config("sub", SubConfig)

        sys.argv = ["test", "--sub.depth", "7"]
        build_config()
        self.assertEqual(config_module.DEFAULT_CONFIG.sub.depth, 7)  # type: ignore[union-attr]

    def test_config_type_remains_correct(self) -> None:
        """After build_config, DEFAULT_CONFIG should still be a Config instance."""
        sys.argv = ["test", "--max-budget", "1.0"]
        build_config()
        self.assertIsInstance(config_module.DEFAULT_CONFIG, Config)


if __name__ == "__main__":
    unittest.main()
