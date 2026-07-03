# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Regression tests for ``_parse_text_based_tool_calls``.

Verifies that when a single model response contains multiple distinct
``{"tool_calls": [...]}`` JSON blocks (e.g. a Bash call followed by a
separate go_to_url call — a real ``cc/opus`` emission pattern), every
call is extracted in order, and that exact duplicates collapse to one
entry.
"""

import json
import unittest

from kiss.core.models.model import _parse_text_based_tool_calls


class TestParseMultipleToolCallBlocks(unittest.TestCase):
    """Verify multi-block / multi-call extraction with deduplication."""

    def test_two_distinct_inline_blocks_both_extracted(self) -> None:
        """Two separate ``{"tool_calls": [...]}`` JSON blocks → both extracted."""
        bash_block = (
            '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        )
        goto_block = (
            '{"tool_calls": [{"name": "go_to_url", "arguments": {"url": "https://x.com"}}]}'
        )
        content = f"First I will list files.\n{bash_block}\nThen visit:\n{goto_block}"

        calls = _parse_text_based_tool_calls(content)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], "Bash")
        self.assertEqual(calls[0]["arguments"], {"command": "ls"})
        self.assertEqual(calls[1]["name"], "go_to_url")
        self.assertEqual(calls[1]["arguments"], {"url": "https://x.com"})

    def test_stray_open_brace_in_prose_before_valid_block(self) -> None:
        """A stray unbalanced ``{`` in prose must not abort the whole scan."""
        content = (
            "Note: use { to start a block in this language.\n"
            '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        )

        calls = _parse_text_based_tool_calls(content)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "Bash")
        self.assertEqual(calls[0]["arguments"], {"command": "ls"})

    def test_two_calls_in_one_block(self) -> None:
        """A single ``tool_calls`` array with two entries → both extracted."""
        content = (
            '{"tool_calls": ['
            '{"name": "Bash", "arguments": {"command": "ls"}},'
            '{"name": "Read", "arguments": {"file_path": "a.py"}}'
            ']}'
        )

        calls = _parse_text_based_tool_calls(content)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], "Bash")
        self.assertEqual(calls[1]["name"], "Read")

    def test_three_identical_blocks_dedup_to_one(self) -> None:
        """Identical (name, args) repetitions collapse to a single call."""
        block = (
            '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        )
        content = f"{block}\n\n{block}\n\n{block}"

        calls = _parse_text_based_tool_calls(content)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "Bash")
        self.assertEqual(calls[0]["arguments"], {"command": "ls"})

    def test_distinct_args_same_name_not_deduped(self) -> None:
        """Same tool name with different arguments → both kept."""
        b1 = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        b2 = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "pwd"}}]}'
        content = f"{b1}\n{b2}"

        calls = _parse_text_based_tool_calls(content)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["arguments"], {"command": "ls"})
        self.assertEqual(calls[1]["arguments"], {"command": "pwd"})

    def test_fenced_json_then_inline(self) -> None:
        """A fenced ```json``` block plus an inline block → both extracted."""
        content = (
            "I'll do two things:\n\n"
            "```json\n"
            '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}\n'
            "```\n\n"
            "Then:\n"
            '{"tool_calls": [{"name": "go_to_url", "arguments": {"url": "https://x.com"}}]}'
        )

        calls = _parse_text_based_tool_calls(content)

        names = [c["name"] for c in calls]
        self.assertIn("Bash", names)
        self.assertIn("go_to_url", names)
        # No spurious duplicates
        self.assertEqual(len(calls), 2)

    def test_unique_ids_per_call(self) -> None:
        """Each extracted call gets a fresh id, even on dedup-survivors."""
        b1 = '{"tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}]}'
        b2 = '{"tool_calls": [{"name": "Read", "arguments": {"file_path": "a"}}]}'
        content = f"{b1}\n{b2}"

        calls = _parse_text_based_tool_calls(content)

        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0]["id"], calls[1]["id"])
        for c in calls:
            self.assertTrue(c["id"].startswith("call_"))

    def test_existing_single_block_still_works(self) -> None:
        """Single-block behavior is preserved (regression sanity)."""
        content = json.dumps(
            {"tool_calls": [{"name": "finish", "arguments": {"success": "True"}}]}
        )
        calls = _parse_text_based_tool_calls(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "finish")


if __name__ == "__main__":
    unittest.main()
