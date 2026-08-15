# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Verify that ``ClaudeCodeModel._build_cli_args`` invokes the ``claude``
CLI in agentic mode, mirroring how ``CodexModel`` invokes ``codex exec``.

The CLI's native tools stay enabled and permission prompts are bypassed
(``--dangerously-skip-permissions``) because KISS is the outer agent and
the user has already authorized KISS to act on their behalf.
``--disable-slash-commands`` and ``--no-session-persistence`` keep each
invocation self-contained.
"""

import unittest

from kiss.core.models.claude_code_model import ClaudeCodeModel


class TestClaudeCodeAgenticFlags(unittest.TestCase):
    """Assert agentic CLI flags are present in every invocation."""

    def test_disable_slash_commands_present(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        args = m._build_cli_args()
        self.assertIn("--disable-slash-commands", args)

    def test_agentic_mode_flags(self) -> None:
        """Native tools stay enabled and permission checks are bypassed."""
        m = ClaudeCodeModel("cc/opus")
        args = m._build_cli_args()
        self.assertIn("--print", args)
        self.assertIn("--no-session-persistence", args)
        self.assertIn("--dangerously-skip-permissions", args)
        self.assertNotIn("--tools", args)

    def test_model_flag_uses_cli_model_alias(self) -> None:
        """The ``cc/`` prefix is stripped before passing to ``--model``."""
        m = ClaudeCodeModel("cc/sonnet")
        args = m._build_cli_args()
        idx = args.index("--model")
        self.assertEqual(args[idx + 1], "sonnet")

    def test_flags_present_with_system_instruction(self) -> None:
        """Agentic flags survive even when a system prompt is configured."""
        m = ClaudeCodeModel("cc/opus", model_config={"system_instruction": "be brief"})
        args = m._build_cli_args()
        self.assertIn("--disable-slash-commands", args)
        self.assertIn("--dangerously-skip-permissions", args)
        self.assertIn("--system-prompt", args)
        idx = args.index("--system-prompt")
        self.assertEqual(args[idx + 1], "be brief")


if __name__ == "__main__":
    unittest.main()
