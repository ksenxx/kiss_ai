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
from kiss.core.models.model import CLI_SYSTEM_PROMPT_HEADER


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

    def test_system_instruction_is_appended_system_prompt(self) -> None:
        """The system prompt is passed as ``--append-system-prompt`` (the CLI
        keeps its own native system prompt), never as ``--system-prompt``
        and never inside the task prompt, which Claude reads as an
        injected fake system prompt."""
        m = ClaudeCodeModel("cc/opus", model_config={"system_instruction": "be brief"})
        args = m._build_cli_args()
        self.assertIn("--disable-slash-commands", args)
        self.assertIn("--dangerously-skip-permissions", args)
        self.assertNotIn("--system-prompt", args)
        self.assertEqual(args[args.index("--append-system-prompt") + 1], "be brief")
        m.initialize("do the task")
        self.assertEqual(m._build_prompt(), "do the task")
        self.assertNotIn(CLI_SYSTEM_PROMPT_HEADER, m._build_prompt())

    def test_no_system_instruction_means_no_append_flag(self) -> None:
        """Without a system instruction the CLI keeps only its own prompt."""
        m = ClaudeCodeModel("cc/opus")
        self.assertNotIn("--append-system-prompt", m._build_cli_args())
        m.initialize("do the task")
        self.assertEqual(m._build_prompt(), "do the task")


if __name__ == "__main__":
    unittest.main()
