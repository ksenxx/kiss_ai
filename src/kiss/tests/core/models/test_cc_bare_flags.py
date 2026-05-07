# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Verify that ``ClaudeCodeModel._build_cli_args`` invokes the ``claude``
CLI in pure single-shot LLM mode.

The Sorcar agent — not the Claude Code CLI — must be the one parsing
tool calls and dispatching them.  ``--bare`` and ``--disable-slash-commands``
disable hooks, LSP, plugin sync, auto-memory, background prefetches,
keychain reads, CLAUDE.md auto-discovery, skills, and agents so the CLI
behaves as a thin wrapper around a single API call.
"""

import unittest

from kiss.core.models.claude_code_model import ClaudeCodeModel


class TestClaudeCodeBareFlags(unittest.TestCase):
    """Assert pure-LLM CLI flags are present in every invocation."""

    def test_bare_and_disable_slash_commands_present(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        args = m._build_cli_args()
        self.assertIn("--bare", args)
        self.assertIn("--disable-slash-commands", args)

    def test_print_and_no_session_persistence_and_tools_disabled(self) -> None:
        """Existing single-shot flags must still be present."""
        m = ClaudeCodeModel("cc/opus")
        args = m._build_cli_args()
        self.assertIn("--print", args)
        self.assertIn("--no-session-persistence", args)
        # ``--tools`` followed by an empty string disables built-in tools.
        self.assertIn("--tools", args)
        idx = args.index("--tools")
        self.assertEqual(args[idx + 1], "")

    def test_model_flag_uses_cli_model_alias(self) -> None:
        """The ``cc/`` prefix is stripped before passing to ``--model``."""
        m = ClaudeCodeModel("cc/sonnet")
        args = m._build_cli_args()
        idx = args.index("--model")
        self.assertEqual(args[idx + 1], "sonnet")

    def test_flags_present_with_system_instruction(self) -> None:
        """``--bare`` survives even when a system prompt is configured."""
        m = ClaudeCodeModel("cc/opus", model_config={"system_instruction": "be brief"})
        args = m._build_cli_args()
        self.assertIn("--bare", args)
        self.assertIn("--disable-slash-commands", args)
        self.assertIn("--system-prompt", args)
        idx = args.index("--system-prompt")
        self.assertEqual(args[idx + 1], "be brief")


if __name__ == "__main__":
    unittest.main()
